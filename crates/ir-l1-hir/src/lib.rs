use cas_store::CasHash;
use ir_l0_cst::OriginId;

/// Canonicalized POSIX-style relative workspace path.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NormPath(pub String);

impl NormPath {
    pub fn new(raw: &str) -> Self {
        let mut normalized = raw.replace('\\', "/");
        while normalized.contains("//") {
            normalized = normalized.replace("//", "/");
        }
        while let Some(stripped) = normalized.strip_prefix("./") {
            normalized = stripped.to_string();
        }
        Self(normalized)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Explicit memory barrier epoch guarding opaque side-effecting operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MemoryBarrier {
    pub epoch: u64,
}

/// Lightweight Provenance Algebra replacing downward vendor field pass-through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Unmodified 1:1 projection from an L0 event; eligible for lossless rehydration.
    Exact(OriginId),
    /// Multiple L0 operations fused or compacted into one canonical semantic op.
    Fused(Vec<OriginId>),
    /// Reused result pulled from content-addressed cache.
    Reused(CasHash),
    /// Newly synthesized node created during optimization or cross-agent lowering.
    Synthesized,
}

/// Strongly typed canonical L1 dialect operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DialectOp {
    UserPrompt { content: CasHash },
    AssistantMessage { content: CasHash },
    FsRead { path: NormPath },
    EditPatch { target: NormPath, diff: CasHash },
    ShellExec { cmd: CasHash, barrier: MemoryBarrier },
    OpaqueToolError { tool_name: String, raw_payload: CasHash },
}

/// A single L1 HIR instruction paired with its provenance lineage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HirOp {
    pub op: DialectOp,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowerError {
    InvalidEvent(String),
    MissingField(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmitError {
    MissingComplement(OriginId),
    MissingCasBlob(CasHash),
    InvalidProvenance(String),
}

/// Asymmetric Lens contract separating pure semantic projection (`get`) from
/// complement-guided rehydration (`put`).
///
/// `-O0` GetPut law: for unmodified `Exact` nodes, `put(get(raw)) == raw`.
pub trait AsymmetricLens {
    type Raw;
    type Complement;

    fn get(&self, raw: &Self::Raw) -> Result<(Vec<HirOp>, Self::Complement), LowerError>;
    fn put(&self, ops: &[HirOp], comp: &Self::Complement) -> Result<Self::Raw, EmitError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_cross_platform_paths() {
        assert_eq!(NormPath::new(".\\src\\\\main.rs").as_str(), "src/main.rs");
        assert_eq!(
            NormPath::new("./crates//ir-l1-hir/src/lib.rs").as_str(),
            "crates/ir-l1-hir/src/lib.rs"
        );
    }
}
