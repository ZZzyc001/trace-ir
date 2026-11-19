use std::collections::HashMap;

use ir_l1_hir::{AsymmetricLens, DialectOp};
use ir_l2_mir::{MirDag, NodeId, ResourceId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    DepOrderViolation { node: NodeId, dep: NodeId },
    BarrierCrossed { node: NodeId, node_epoch: u64, barrier_epoch: u64 },
    DepEpochInversion { node: NodeId, dep: NodeId },
    NonMonotonicToken { res: ResourceId, prev: u64, actual: u64 },
}

/// Verifies L2 MIR structural and MemorySSA invariants:
/// 1. Dependencies strictly respect topological node ordering.
/// 2. Opaque `ShellExec` ops bind to their exact `MemoryBarrier` epoch and never reorder across barriers.
/// 3. `StateToken` write versions are strictly monotonic (`prev + 1`) per `ResourceId`.
pub fn verify_l2_invariants(dag: &MirDag) -> Result<(), VerifyError> {
    let mut last_ver: HashMap<ResourceId, u64> = HashMap::new();

    for node in &dag.nodes {
        for &dep in &node.deps {
            if dep.0 >= node.id.0 || dep.0 >= dag.nodes.len() {
                return Err(VerifyError::DepOrderViolation {
                    node: node.id,
                    dep,
                });
            }
            if dag.nodes[dep.0].epoch > node.epoch {
                return Err(VerifyError::DepEpochInversion {
                    node: node.id,
                    dep,
                });
            }
        }

        if let DialectOp::ShellExec { barrier, .. } = &node.hir.op {
            if node.epoch != barrier.epoch {
                return Err(VerifyError::BarrierCrossed {
                    node: node.id,
                    node_epoch: node.epoch,
                    barrier_epoch: barrier.epoch,
                });
            }
        }

        for tok in &node.writes {
            if let Some(&prev) = last_ver.get(&tok.res) {
                if tok.ver != prev + 1 {
                    return Err(VerifyError::NonMonotonicToken {
                        res: tok.res.clone(),
                        prev,
                        actual: tok.ver,
                    });
                }
            }
            last_ver.insert(tok.res.clone(), tok.ver);
        }
    }

    Ok(())
}

/// Generic `-O0` GetPut law conformance check for any `AsymmetricLens`.
pub fn verify_o0_getput<L>(lens: &L, raw: &L::Raw) -> Result<(), String>
where
    L: AsymmetricLens,
    L::Raw: PartialEq + core::fmt::Debug,
{
    let (hir, comp) = lens
        .get(raw)
        .map_err(|e| format!("lowering failed: {e:?}"))?;
    let restored = lens
        .put(&hir, &comp)
        .map_err(|e| format!("rehydration failed: {e:?}"))?;
    if &restored != raw {
        return Err(format!(
            "GetPut violation:\n  expected: {raw:?}\n  restored: {restored:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cas_store::CasStore;
    use ir_l0_cst::ComplementTable;
    use ir_l1_hir::MemoryBarrier;
    use ir_l2_mir::MirDag;
    use lens_astra::{AstraLens, RawAstraPayload};
    use lens_claude::{ClaudeLens, RawClaudeEvent, RawClaudeKind, RawClaudeTrace};
    use proptest::prelude::*;

    use super::*;

    fn arb_claude_kind() -> impl Strategy<Value = RawClaudeKind> {
        prop_oneof![
            ".{0,40}".prop_map(RawClaudeKind::UserText),
            ".{0,40}".prop_map(RawClaudeKind::AssistantText),
            "[a-z0-9_./\\\\]{1,24}".prop_map(|raw_path| RawClaudeKind::ToolUseRead { raw_path }),
            ("[a-z0-9_./\\\\]{1,24}", ".{0,32}")
                .prop_map(|(raw_path, diff)| RawClaudeKind::ToolUseEdit { raw_path, diff }),
            "[a-z0-9 _-]{1,32}".prop_map(|command| RawClaudeKind::ToolUseBash { command }),
            ("[a-z_]{1,16}", ".{0,32}").prop_map(|(tool_name, raw_chunk)| {
                RawClaudeKind::MalformedDelta {
                    tool_name,
                    raw_chunk,
                }
            }),
        ]
    }

    fn arb_claude_trace() -> impl Strategy<Value = RawClaudeTrace> {
        prop::collection::vec(
            (
                any::<u64>(),
                prop::option::of("[a-z0-9_]{4,16}"),
                "[a-z0-9-]{4,16}",
                arb_claude_kind(),
            ),
            0..24,
        )
        .prop_map(|items| {
            let events = items
                .into_iter()
                .enumerate()
                .map(
                    |(idx, (timestamp_ms, reasoning_signature, model_tag, kind))| RawClaudeEvent {
                        event_id: idx as u64 + 1,
                        timestamp_ms,
                        reasoning_signature,
                        model_tag,
                        kind,
                    },
                )
                .collect();
            RawClaudeTrace { events }
        })
    }

    proptest! {
        #[test]
        fn prop_o0_claude_getput_lossless(raw in arb_claude_trace()) {
            let lens = ClaudeLens::new_in_memory();
            prop_assert!(verify_o0_getput(&lens, &raw).is_ok());
        }

        #[test]
        fn prop_l2_raised_and_o2_dags_satisfy_invariants(raw in arb_claude_trace()) {
            let lens = ClaudeLens::new_in_memory();
            let (hir, _) = lens.get(&raw).unwrap();
            let dag = MirDag::raise_from_hir(&hir);
            prop_assert!(verify_l2_invariants(&dag).is_ok());
            let opt = dag.optimize_o2();
            prop_assert!(verify_l2_invariants(&opt).is_ok());
        }
    }

    #[test]
    fn rejects_injected_barrier_violation() {
        let lens = ClaudeLens::new_in_memory();
        let raw = RawClaudeTrace {
            events: vec![RawClaudeEvent {
                event_id: 1,
                timestamp_ms: 100,
                reasoning_signature: None,
                model_tag: "claude".to_string(),
                kind: RawClaudeKind::ToolUseBash {
                    command: "cargo test".to_string(),
                },
            }],
        };
        let (hir, _) = lens.get(&raw).unwrap();
        let mut dag = MirDag::raise_from_hir(&hir);
        if let DialectOp::ShellExec { barrier, .. } = &mut dag.nodes[0].hir.op {
            *barrier = MemoryBarrier { epoch: 99 };
        }
        assert!(matches!(
            verify_l2_invariants(&dag),
            Err(VerifyError::BarrierCrossed { .. })
        ));
    }

    #[test]
    fn claude_to_astra_handoff_with_o2_compaction() {
        let cas = Arc::new(CasStore::new());
        let claude = ClaudeLens::new(Arc::clone(&cas));
        let astra = AstraLens::new(Arc::clone(&cas));

        let raw_claude = RawClaudeTrace {
            events: vec![
                RawClaudeEvent {
                    event_id: 1,
                    timestamp_ms: 1000,
                    reasoning_signature: None,
                    model_tag: "claude-opus".to_string(),
                    kind: RawClaudeKind::UserText("Fix borrow checker error".to_string()),
                },
                RawClaudeEvent {
                    event_id: 2,
                    timestamp_ms: 1001,
                    reasoning_signature: Some("sig_1".to_string()),
                    model_tag: "claude-opus".to_string(),
                    kind: RawClaudeKind::ToolUseRead {
                        raw_path: "./src/lib.rs".to_string(),
                    },
                },
                RawClaudeEvent {
                    event_id: 3,
                    timestamp_ms: 1002,
                    reasoning_signature: Some("sig_err".to_string()),
                    model_tag: "claude-opus".to_string(),
                    kind: RawClaudeKind::MalformedDelta {
                        tool_name: "edit".to_string(),
                        raw_chunk: "broken json".to_string(),
                    },
                },
                RawClaudeEvent {
                    event_id: 4,
                    timestamp_ms: 1003,
                    reasoning_signature: Some("sig_2".to_string()),
                    model_tag: "claude-opus".to_string(),
                    kind: RawClaudeKind::ToolUseRead {
                        raw_path: "src//lib.rs".to_string(),
                    },
                },
            ],
        };

        let (hir, _) = claude.get(&raw_claude).unwrap();
        let opt_dag = MirDag::raise_from_hir(&hir).optimize_o2();
        assert!(verify_l2_invariants(&opt_dag).is_ok());
        // 4 turns -> 2 turns (pruned malformed retry + folded duplicate read)
        assert_eq!(opt_dag.nodes.len(), 2);

        let compacted_hir: Vec<_> = opt_dag.nodes.into_iter().map(|n| n.hir).collect();
        let astra_trace = astra
            .put(&compacted_hir, &ComplementTable::new())
            .expect("cross-agent handoff into Astra should succeed");
        assert_eq!(astra_trace.frames.len(), 2);
        assert_eq!(
            astra_trace.frames[1].payload,
            RawAstraPayload::ReadFile {
                path: "src/lib.rs".to_string()
            }
        );
    }
}
