use std::collections::{HashMap, HashSet};

use ir_l0_cst::OriginId;
use ir_l1_hir::{DialectOp, HirOp, Provenance};

/// Logical external resource identifier (e.g., normalized file path).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(pub String);

/// MemorySSA-style versioned state token (e.g., `a.py@v0 -> a.py@v1`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateToken {
    pub res: ResourceId,
    pub ver: u64,
}

/// Dense node index inside a `MirDag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Effect-annotated MIR node inside the L2 dependency DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MirNode {
    pub id: NodeId,
    pub epoch: u64,
    pub hir: HirOp,
    pub reads: Vec<StateToken>,
    pub writes: Vec<StateToken>,
    pub deps: Vec<NodeId>,
}

/// L2 Effect Dependency Graph supporting MemorySSA analysis, `-O2` optimization,
/// and barrier-safe parallel list scheduling.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MirDag {
    pub nodes: Vec<MirNode>,
}

impl MirDag {
    /// Raises a linear L1 HIR trace into an explicit L2 effect dependency DAG.
    pub fn raise_from_hir(ops: &[HirOp]) -> Self {
        let mut nodes: Vec<MirNode> = Vec::with_capacity(ops.len());
        let mut current_epoch: u64 = 0;
        let mut last_barrier: Option<NodeId> = None;
        let mut last_writer: HashMap<ResourceId, (u64, NodeId)> = HashMap::new();

        for hir in ops {
            let id = NodeId(nodes.len());
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            let mut deps = Vec::new();

            if let Some(b) = last_barrier {
                deps.push(b);
            }

            let node_epoch = match &hir.op {
                DialectOp::FsRead { path } => {
                    let res = ResourceId(path.as_str().to_string());
                    let ver = if let Some(&(v, writer_id)) = last_writer.get(&res) {
                        if !deps.contains(&writer_id) {
                            deps.push(writer_id);
                        }
                        v
                    } else {
                        0
                    };
                    reads.push(StateToken { res, ver });
                    current_epoch
                }
                DialectOp::EditPatch { target, .. } => {
                    let res = ResourceId(target.as_str().to_string());
                    let prev_ver = if let Some(&(v, writer_id)) = last_writer.get(&res) {
                        if !deps.contains(&writer_id) {
                            deps.push(writer_id);
                        }
                        v
                    } else {
                        0
                    };
                    let next_ver = prev_ver + 1;
                    reads.push(StateToken {
                        res: res.clone(),
                        ver: prev_ver,
                    });
                    writes.push(StateToken {
                        res: res.clone(),
                        ver: next_ver,
                    });
                    last_writer.insert(res, (next_ver, id));
                    current_epoch
                }
                DialectOp::ShellExec { barrier, .. } => {
                    let target_epoch = barrier.epoch.max(current_epoch + 1);
                    for prev in &nodes {
                        if prev.epoch < target_epoch && !deps.contains(&prev.id) {
                            deps.push(prev.id);
                        }
                    }
                    current_epoch = target_epoch;
                    last_barrier = Some(id);
                    target_epoch
                }
                DialectOp::UserPrompt { .. }
                | DialectOp::AssistantMessage { .. }
                | DialectOp::OpaqueToolError { .. } => current_epoch,
            };

            let mut adjusted_hir = hir.clone();
            if let DialectOp::ShellExec { barrier, .. } = &mut adjusted_hir.op {
                barrier.epoch = node_epoch;
            }

            nodes.push(MirNode {
                id,
                epoch: node_epoch,
                hir: adjusted_hir,
                reads,
                writes,
                deps,
            });
        }

        Self { nodes }
    }

    /// Runs `-O2` trace optimizations while preserving all `UserPrompt` anchors and `MemoryBarrier` epochs:
    /// 1. Dead Code Elimination (DCE): prunes non-barrier `OpaqueToolError` failed retry turns.
    /// 2. Common Subexpression Elimination (CSE): folds duplicate `FsRead` on the same `(ResourceId, ver, epoch)`.
    pub fn optimize_o2(&self) -> Self {
        let mut kept_ops: Vec<HirOp> = Vec::with_capacity(self.nodes.len());
        let mut seen_reads: HashMap<(u64, ResourceId, u64), usize> = HashMap::new();

        for node in &self.nodes {
            match &node.hir.op {
                DialectOp::OpaqueToolError { .. } => {
                    continue;
                }
                DialectOp::FsRead { path } => {
                    let res = ResourceId(path.as_str().to_string());
                    let ver = node.reads.first().map(|t| t.ver).unwrap_or(0);
                    let key = (node.epoch, res, ver);
                    if let Some(&first_idx) = seen_reads.get(&key) {
                        let merged =
                            merge_provenance(&kept_ops[first_idx].provenance, &node.hir.provenance);
                        kept_ops[first_idx].provenance = merged;
                        continue;
                    }
                    let idx = kept_ops.len();
                    seen_reads.insert(key, idx);
                    kept_ops.push(node.hir.clone());
                }
                _ => kept_ops.push(node.hir.clone()),
            }
        }

        Self::raise_from_hir(&kept_ops)
    }

    /// List scheduler that groups independent nodes in the same barrier epoch
    /// into concurrent execution batches (`parallel_tool_calls`).
    pub fn schedule_parallel_batches(&self) -> Vec<Vec<NodeId>> {
        let mut scheduled: HashSet<NodeId> = HashSet::new();
        let mut batches: Vec<Vec<NodeId>> = Vec::new();

        while scheduled.len() < self.nodes.len() {
            let mut ready: Vec<NodeId> = Vec::new();
            let mut batch_epoch: Option<u64> = None;

            for node in &self.nodes {
                if scheduled.contains(&node.id) {
                    continue;
                }
                let deps_done = node.deps.iter().all(|d| scheduled.contains(d));
                if !deps_done {
                    continue;
                }
                match batch_epoch {
                    None => {
                        batch_epoch = Some(node.epoch);
                        ready.push(node.id);
                    }
                    Some(ep) if ep == node.epoch => {
                        ready.push(node.id);
                    }
                    Some(_) => {}
                }
            }

            if ready.is_empty() {
                break;
            }
            for &id in &ready {
                scheduled.insert(id);
            }
            batches.push(ready);
        }

        batches
    }
}

fn collect_origins(prov: &Provenance) -> Vec<OriginId> {
    match prov {
        Provenance::Exact(id) => vec![*id],
        Provenance::Fused(ids) => ids.clone(),
        Provenance::Reused(_) | Provenance::Synthesized => Vec::new(),
    }
}

fn merge_provenance(a: &Provenance, b: &Provenance) -> Provenance {
    let mut ids = collect_origins(a);
    ids.extend(collect_origins(b));
    if ids.is_empty() {
        Provenance::Synthesized
    } else {
        Provenance::Fused(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_store::CasHash;
    use ir_l1_hir::{MemoryBarrier, NormPath};

    #[test]
    fn o2_prunes_errors_folds_reads_and_preserves_barriers() {
        let ops = vec![
            HirOp {
                op: DialectOp::FsRead {
                    path: NormPath::new("src/lib.rs"),
                },
                provenance: Provenance::Exact(OriginId(1)),
            },
            HirOp {
                op: DialectOp::OpaqueToolError {
                    tool_name: "edit".to_string(),
                    raw_payload: CasHash::compute(b"malformed json delta"),
                },
                provenance: Provenance::Exact(OriginId(2)),
            },
            HirOp {
                op: DialectOp::FsRead {
                    path: NormPath::new("./src/lib.rs"),
                },
                provenance: Provenance::Exact(OriginId(3)),
            },
            HirOp {
                op: DialectOp::ShellExec {
                    cmd: CasHash::compute(b"cargo test"),
                    barrier: MemoryBarrier { epoch: 1 },
                },
                provenance: Provenance::Exact(OriginId(4)),
            },
            HirOp {
                op: DialectOp::FsRead {
                    path: NormPath::new("src/lib.rs"),
                },
                provenance: Provenance::Exact(OriginId(5)),
            },
        ];

        let dag = MirDag::raise_from_hir(&ops);
        let opt = dag.optimize_o2();
        assert_eq!(opt.nodes.len(), 3);
        assert_eq!(
            opt.nodes[0].hir.provenance,
            Provenance::Fused(vec![OriginId(1), OriginId(3)])
        );
        assert_eq!(opt.nodes[1].epoch, 1);
        assert_eq!(opt.nodes[2].epoch, 1);
    }

    #[test]
    fn schedules_independent_reads_in_parallel_batch() {
        let ops = vec![
            HirOp {
                op: DialectOp::FsRead {
                    path: NormPath::new("a.rs"),
                },
                provenance: Provenance::Exact(OriginId(1)),
            },
            HirOp {
                op: DialectOp::FsRead {
                    path: NormPath::new("b.rs"),
                },
                provenance: Provenance::Exact(OriginId(2)),
            },
            HirOp {
                op: DialectOp::ShellExec {
                    cmd: CasHash::compute(b"cargo check"),
                    barrier: MemoryBarrier { epoch: 1 },
                },
                provenance: Provenance::Exact(OriginId(3)),
            },
        ];
        let dag = MirDag::raise_from_hir(&ops);
        let batches = dag.schedule_parallel_batches();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![NodeId(0), NodeId(1)]);
        assert_eq!(batches[1], vec![NodeId(2)]);
    }
}
