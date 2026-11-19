use std::collections::HashMap;
use std::sync::RwLock;

/// 32-byte content-addressed digest for immutable trace payloads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CasHash(pub [u8; 32]);

impl CasHash {
    /// Deterministic 256-bit digest using four independent 64-bit FNV-1a lanes.
    /// Designed to be swapped for BLAKE3 behind a feature flag without changing IR types.
    pub fn compute(bytes: &[u8]) -> Self {
        const SEEDS: [u64; 4] = [
            0xcbf29ce484222325,
            0x84222325cbf29ce4,
            0xaf63dc4c8601ec8c,
            0x100000001b3f7a59,
        ];
        const PRIMES: [u64; 4] = [
            0x00000100000001b3,
            0x00000100000001d9,
            0x0000010000000233,
            0x0000010000000265,
        ];

        let mut out = [0u8; 32];
        for lane in 0..4 {
            let mut h = SEEDS[lane] ^ (bytes.len() as u64).wrapping_mul(PRIMES[lane]);
            for (idx, &b) in bytes.iter().enumerate() {
                h ^= (b as u64).wrapping_add((idx as u64).wrapping_mul((lane as u64) + 1));
                h = h.wrapping_mul(PRIMES[lane]);
            }
            out[lane * 8..(lane + 1) * 8].copy_from_slice(&h.to_le_bytes());
        }
        Self(out)
    }
}

/// Global content-addressed constant pool for immutable prompts, diffs, and tool outputs.
#[derive(Debug, Default)]
pub struct CasStore {
    blobs: RwLock<HashMap<CasHash, Vec<u8>>>,
}

impl CasStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&self, data: &[u8]) -> CasHash {
        let hash = CasHash::compute(data);
        let mut guard = self.blobs.write().expect("cas-store lock poisoned");
        guard.entry(hash).or_insert_with(|| data.to_vec());
        hash
    }

    pub fn put_str(&self, text: &str) -> CasHash {
        self.put(text.as_bytes())
    }

    pub fn get(&self, hash: CasHash) -> Option<Vec<u8>> {
        let guard = self.blobs.read().expect("cas-store lock poisoned");
        guard.get(&hash).cloned()
    }

    pub fn get_str(&self, hash: CasHash) -> Option<String> {
        self.get(hash)
            .and_then(|bytes| String::from_utf8(bytes).ok())
    }

    pub fn len(&self) -> usize {
        self.blobs.read().expect("cas-store lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduplicates_identical_payloads() {
        let cas = CasStore::new();
        let h1 = cas.put_str("system prompt payload");
        let h2 = cas.put_str("system prompt payload");
        assert_eq!(h1, h2);
        assert_eq!(cas.len(), 1);
        assert_eq!(cas.get_str(h1).as_deref(), Some("system prompt payload"));
    }

    #[test]
    fn distinguishes_different_payloads() {
        let cas = CasStore::new();
        let h1 = cas.put_str("turn-1");
        let h2 = cas.put_str("turn-2");
        assert_ne!(h1, h2);
        assert_eq!(cas.len(), 2);
    }
}
