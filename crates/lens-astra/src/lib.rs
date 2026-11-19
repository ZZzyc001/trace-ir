use std::sync::Arc;

use cas_store::{CasHash, CasStore};
use ir_l0_cst::{ComplementEntry, ComplementTable, OriginId};
use ir_l1_hir::{
    AsymmetricLens, DialectOp, EmitError, HirOp, LowerError, MemoryBarrier, NormPath, Provenance,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawAstraPayload {
    UserMessage(String),
    ModelOutput(String),
    ReadFile { path: String },
    ApplyPatch { path: String, patch: String },
    ExecShell { script: String },
    ToolFault { tool: String, detail: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawAstraFrame {
    pub frame_id: u64,
    pub trace_span_id: String,
    pub workspace_root: String,
    pub payload: RawAstraPayload,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RawAstraTrace {
    pub frames: Vec<RawAstraFrame>,
}

/// Asymmetric Lens bridging Astra agent frames and canonical L1 HIR.
pub struct AstraLens {
    cas: Arc<CasStore>,
}

impl AstraLens {
    pub fn new(cas: Arc<CasStore>) -> Self {
        Self { cas }
    }

    pub fn new_in_memory() -> Self {
        Self::new(Arc::new(CasStore::new()))
    }
}

impl AsymmetricLens for AstraLens {
    type Raw = RawAstraTrace;
    type Complement = ComplementTable;

    fn get(&self, raw: &Self::Raw) -> Result<(Vec<HirOp>, Self::Complement), LowerError> {
        let mut comp = ComplementTable::new();
        let mut ops = Vec::with_capacity(raw.frames.len());
        let mut next_barrier_epoch: u64 = 1;

        for frame in &raw.frames {
            let origin = OriginId(frame.frame_id);
            let mut raw_attributes = vec![
                ("trace_span_id".to_string(), frame.trace_span_id.clone()),
                ("workspace_root".to_string(), frame.workspace_root.clone()),
            ];

            let dialect_op = match &frame.payload {
                RawAstraPayload::UserMessage(text) => DialectOp::UserPrompt {
                    content: self.cas.put_str(text),
                },
                RawAstraPayload::ModelOutput(text) => DialectOp::AssistantMessage {
                    content: self.cas.put_str(text),
                },
                RawAstraPayload::ReadFile { path } => {
                    raw_attributes.push(("raw_path".to_string(), path.clone()));
                    DialectOp::FsRead {
                        path: NormPath::new(path),
                    }
                }
                RawAstraPayload::ApplyPatch { path, patch } => {
                    raw_attributes.push(("raw_path".to_string(), path.clone()));
                    DialectOp::EditPatch {
                        target: NormPath::new(path),
                        diff: self.cas.put_str(patch),
                    }
                }
                RawAstraPayload::ExecShell { script } => {
                    let barrier = MemoryBarrier {
                        epoch: next_barrier_epoch,
                    };
                    next_barrier_epoch += 1;
                    DialectOp::ShellExec {
                        cmd: self.cas.put_str(script),
                        barrier,
                    }
                }
                RawAstraPayload::ToolFault { tool, detail } => DialectOp::OpaqueToolError {
                    tool_name: tool.clone(),
                    raw_payload: self.cas.put_str(detail),
                },
            };

            comp.insert(
                origin,
                ComplementEntry {
                    timestamp_ms: 0,
                    reasoning_signature: None,
                    model_tag: "astra".to_string(),
                    vendor_envelope_hash: CasHash::compute(frame.trace_span_id.as_bytes()),
                    raw_attributes,
                },
            );

            ops.push(HirOp {
                op: dialect_op,
                provenance: Provenance::Exact(origin),
            });
        }

        Ok((ops, comp))
    }

    fn put(&self, ops: &[HirOp], comp: &Self::Complement) -> Result<Self::Raw, EmitError> {
        let mut frames = Vec::with_capacity(ops.len());

        for (idx, hir) in ops.iter().enumerate() {
            let (frame_id, entry_opt) = match &hir.provenance {
                Provenance::Exact(origin) => (origin.0, comp.get(*origin)),
                Provenance::Fused(origins) => {
                    let first = origins.first().copied();
                    (
                        first.map(|o| o.0).unwrap_or(idx as u64),
                        first.and_then(|o| comp.get(o)),
                    )
                }
                Provenance::Reused(_) | Provenance::Synthesized => (idx as u64, None),
            };

            let attr = |key: &str| -> Option<String> {
                entry_opt.and_then(|e| {
                    e.raw_attributes
                        .iter()
                        .find(|(k, _)| k == key)
                        .map(|(_, v)| v.clone())
                })
            };

            let trace_span_id = attr("trace_span_id")
                .unwrap_or_else(|| format!("handoff-span-{idx}"));
            let workspace_root = attr("workspace_root").unwrap_or_else(|| "/workspace".to_string());
            let raw_path_attr = attr("raw_path");

            let payload = match &hir.op {
                DialectOp::UserPrompt { content } => RawAstraPayload::UserMessage(
                    self.cas
                        .get_str(*content)
                        .ok_or(EmitError::MissingCasBlob(*content))?,
                ),
                DialectOp::AssistantMessage { content } => RawAstraPayload::ModelOutput(
                    self.cas
                        .get_str(*content)
                        .ok_or(EmitError::MissingCasBlob(*content))?,
                ),
                DialectOp::FsRead { path } => RawAstraPayload::ReadFile {
                    path: raw_path_attr.unwrap_or_else(|| path.as_str().to_string()),
                },
                DialectOp::EditPatch { target, diff } => RawAstraPayload::ApplyPatch {
                    path: raw_path_attr.unwrap_or_else(|| target.as_str().to_string()),
                    patch: self
                        .cas
                        .get_str(*diff)
                        .ok_or(EmitError::MissingCasBlob(*diff))?,
                },
                DialectOp::ShellExec { cmd, .. } => RawAstraPayload::ExecShell {
                    script: self
                        .cas
                        .get_str(*cmd)
                        .ok_or(EmitError::MissingCasBlob(*cmd))?,
                },
                DialectOp::OpaqueToolError {
                    tool_name,
                    raw_payload,
                } => RawAstraPayload::ToolFault {
                    tool: tool_name.clone(),
                    detail: self
                        .cas
                        .get_str(*raw_payload)
                        .ok_or(EmitError::MissingCasBlob(*raw_payload))?,
                },
            };

            frames.push(RawAstraFrame {
                frame_id,
                trace_span_id,
                workspace_root,
                payload,
            });
        }

        Ok(RawAstraTrace { frames })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satisfies_astra_o0_getput_law() {
        let lens = AstraLens::new_in_memory();
        let raw = RawAstraTrace {
            frames: vec![
                RawAstraFrame {
                    frame_id: 10,
                    trace_span_id: "span-001".to_string(),
                    workspace_root: "/repo".to_string(),
                    payload: RawAstraPayload::UserMessage("Verify L2 DAG".to_string()),
                },
                RawAstraFrame {
                    frame_id: 11,
                    trace_span_id: "span-002".to_string(),
                    workspace_root: "/repo".to_string(),
                    payload: RawAstraPayload::ReadFile {
                        path: ".\\src\\main.rs".to_string(),
                    },
                },
            ],
        };

        let (hir, comp) = lens.get(&raw).expect("lowering should succeed");
        let restored = lens.put(&hir, &comp).expect("rehydration should succeed");
        assert_eq!(restored, raw);
    }
}
