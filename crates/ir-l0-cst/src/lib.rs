use std::collections::HashMap;

use cas_store::CasHash;

/// Stable identifier referencing a single raw event in the source trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OriginId(pub u64);

/// Reference into a local `ComplementTable` at the L0 boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComplementId(pub u64);

/// Supported source/target agent session ecosystems at the L0 boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VendorKind {
    Claude,
    Astra,
    Codex,
    Devin,
}

/// Lossless L0 envelope node (Red-Green CST style).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L0Node {
    pub origin: OriginId,
    pub vendor: VendorKind,
    pub payload: CasHash,
    pub complement: ComplementId,
}

/// Stripped vendor-specific metadata retained strictly at the L0 boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComplementEntry {
    pub timestamp_ms: u64,
    pub reasoning_signature: Option<String>,
    pub model_tag: String,
    pub vendor_envelope_hash: CasHash,
    pub raw_attributes: Vec<(String, String)>,
}

/// Asymmetric Lens side-table holding vendor complements so private fields never leak into L1/L2.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComplementTable {
    entries: HashMap<OriginId, ComplementEntry>,
}

impl ComplementTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, origin: OriginId, entry: ComplementEntry) {
        self.entries.insert(origin, entry);
    }

    pub fn get(&self, origin: OriginId) -> Option<&ComplementEntry> {
        self.entries.get(&origin)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_retrieves_vendor_complement() {
        let mut table = ComplementTable::new();
        let origin = OriginId(42);
        let entry = ComplementEntry {
            timestamp_ms: 1_795_111_616,
            reasoning_signature: Some("sig_abc123".to_string()),
            model_tag: "claude-opus".to_string(),
            vendor_envelope_hash: CasHash::compute(b"raw-envelope"),
            raw_attributes: vec![("stream_seq".to_string(), "7".to_string())],
        };
        table.insert(origin, entry.clone());
        assert_eq!(table.get(origin), Some(&entry));
    }
}
