use std::sync::Arc;

use cas_store::{CasHash, CasStore};
use ir_l0_cst::{ComplementEntry, ComplementTable, OriginId};
use ir_l1_hir::{
    AsymmetricLens, DialectOp, EmitError, HirOp, LowerError, MemoryBarrier, NormPath, Provenance,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawClaudeKind {
    UserText(String),
    AssistantText(String),
    ToolUseRead { raw_path: String },
    ToolUseEdit { raw_path: String, diff: String },
    ToolUseBash { command: String },
    MalformedDelta { tool_name: String, raw_chunk: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawClaudeEvent {
    pub event_id: u64,
    pub timestamp_ms: u64,
    pub reasoning_signature: Option<String>,
    pub model_tag: String,
    pub kind: RawClaudeKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RawClaudeTrace {
    pub events: Vec<RawClaudeEvent>,
}

/// Asymmetric Lens bridging Claude session traces and canonical L1 HIR.
pub struct ClaudeLens {
    cas: Arc<CasStore>,
}

impl ClaudeLens {
    pub fn new(cas: Arc<CasStore>) -> Self {
        Self { cas }
    }

    pub fn new_in_memory() -> Self {
        Self::new(Arc::new(CasStore::new()))
    }

    pub fn cas(&self) -> &Arc<CasStore> {
        &self.cas
    }
}

impl AsymmetricLens for ClaudeLens {
    type Raw = RawClaudeTrace;
    type Complement = ComplementTable;

    fn get(&self, raw: &Self::Raw) -> Result<(Vec<HirOp>, Self::Complement), LowerError> {
        let mut comp = ComplementTable::new();
        let mut ops = Vec::with_capacity(raw.events.len());
        let mut next_barrier_epoch: u64 = 1;

        for ev in &raw.events {
            let origin = OriginId(ev.event_id);
            let mut raw_attributes = Vec::new();

            let dialect_op = match &ev.kind {
                RawClaudeKind::UserText(text) => DialectOp::UserPrompt {
                    content: self.cas.put_str(text),
                },
                RawClaudeKind::AssistantText(text) => DialectOp::AssistantMessage {
                    content: self.cas.put_str(text),
                },
                RawClaudeKind::ToolUseRead { raw_path } => {
                    raw_attributes.push(("raw_path".to_string(), raw_path.clone()));
                    DialectOp::FsRead {
                        path: NormPath::new(raw_path),
                    }
                }
                RawClaudeKind::ToolUseEdit { raw_path, diff } => {
                    raw_attributes.push(("raw_path".to_string(), raw_path.clone()));
                    DialectOp::EditPatch {
                        target: NormPath::new(raw_path),
                        diff: self.cas.put_str(diff),
                    }
                }
                RawClaudeKind::ToolUseBash { command } => {
                    let barrier = MemoryBarrier {
                        epoch: next_barrier_epoch,
                    };
                    next_barrier_epoch += 1;
                    DialectOp::ShellExec {
                        cmd: self.cas.put_str(command),
                        barrier,
                    }
                }
                RawClaudeKind::MalformedDelta {
                    tool_name,
                    raw_chunk,
                } => DialectOp::OpaqueToolError {
                    tool_name: tool_name.clone(),
                    raw_payload: self.cas.put_str(raw_chunk),
                },
            };

            let envelope_digest = CasHash::compute(
                format!("claude:{}:{}", ev.event_id, ev.model_tag).as_bytes(),
            );
            comp.insert(
                origin,
                ComplementEntry {
                    timestamp_ms: ev.timestamp_ms,
                    reasoning_signature: ev.reasoning_signature.clone(),
                    model_tag: ev.model_tag.clone(),
                    vendor_envelope_hash: envelope_digest,
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
        let mut events = Vec::with_capacity(ops.len());

        for (idx, hir) in ops.iter().enumerate() {
            let (event_id, entry_opt) = match &hir.provenance {
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

            let (timestamp_ms, reasoning_signature, model_tag, raw_path_attr) =
                if let Some(entry) = entry_opt {
                    let rp = entry
                        .raw_attributes
                        .iter()
                        .find(|(k, _)| k == "raw_path")
                        .map(|(_, v)| v.clone());
                    (
                        entry.timestamp_ms,
                        entry.reasoning_signature.clone(),
                        entry.model_tag.clone(),
                        rp,
                    )
                } else {
                    (0, None, "claude-synthesized".to_string(), None)
                };

            let kind = match &hir.op {
                DialectOp::UserPrompt { content } => RawClaudeKind::UserText(
                    self.cas
                        .get_str(*content)
                        .ok_or(EmitError::MissingCasBlob(*content))?,
                ),
                DialectOp::AssistantMessage { content } => RawClaudeKind::AssistantText(
                    self.cas
                        .get_str(*content)
                        .ok_or(EmitError::MissingCasBlob(*content))?,
                ),
                DialectOp::FsRead { path } => RawClaudeKind::ToolUseRead {
                    raw_path: raw_path_attr.unwrap_or_else(|| path.as_str().to_string()),
                },
                DialectOp::EditPatch { target, diff } => RawClaudeKind::ToolUseEdit {
                    raw_path: raw_path_attr.unwrap_or_else(|| target.as_str().to_string()),
                    diff: self
                        .cas
                        .get_str(*diff)
                        .ok_or(EmitError::MissingCasBlob(*diff))?,
                },
                DialectOp::ShellExec { cmd, .. } => RawClaudeKind::ToolUseBash {
                    command: self
                        .cas
                        .get_str(*cmd)
                        .ok_or(EmitError::MissingCasBlob(*cmd))?,
                },
                DialectOp::OpaqueToolError {
                    tool_name,
                    raw_payload,
                } => RawClaudeKind::MalformedDelta {
                    tool_name: tool_name.clone(),
                    raw_chunk: self
                        .cas
                        .get_str(*raw_payload)
                        .ok_or(EmitError::MissingCasBlob(*raw_payload))?,
                },
            };

            events.push(RawClaudeEvent {
                event_id,
                timestamp_ms,
                reasoning_signature,
                model_tag,
                kind,
            });
        }

        Ok(RawClaudeTrace { events })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satisfies_o0_getput_lossless_law() {
        let lens = ClaudeLens::new_in_memory();
        let raw = RawClaudeTrace {
            events: vec![
                RawClaudeEvent {
                    event_id: 1,
                    timestamp_ms: 1_795_111_000,
                    reasoning_signature: None,
                    model_tag: "claude-opus-4".to_string(),
                    kind: RawClaudeKind::UserText("Implement TraceIR L0 lens".to_string()),
                },
                RawClaudeEvent {
                    event_id: 2,
                    timestamp_ms: 1_795_111_120,
                    reasoning_signature: Some("rsig_9f8e7d".to_string()),
                    model_tag: "claude-opus-4".to_string(),
                    kind: RawClaudeKind::ToolUseRead {
                        raw_path: "./crates//ir-l1-hir/src/lib.rs".to_string(),
                    },
                },
                RawClaudeEvent {
                    event_id: 3,
                    timestamp_ms: 1_795_111_240,
                    reasoning_signature: Some("rsig_malformed".to_string()),
                    model_tag: "claude-opus-4".to_string(),
                    kind: RawClaudeKind::MalformedDelta {
                        tool_name: "edit_file".to_string(),
                        raw_chunk: "{\"path\": \"unterminated".to_string(),
                    },
                },
            ],
        };

        let (hir, comp) = lens.get(&raw).expect("lowering should succeed");
        let restored = lens.put(&hir, &comp).expect("rehydration should succeed");
        assert_eq!(restored, raw);
    }
}
