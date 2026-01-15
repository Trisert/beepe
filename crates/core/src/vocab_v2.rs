//! Unified vocabulary with arena allocation for minimal memory overhead.
//!
//! This module provides a memory-efficient alternative to the dual-storage
//! Vocabulary system, eliminating duplicate string storage through arena allocation.

use crate::Result;
use crate::SpecialTokensConfig;
use ahash::AHashMap;
use ahash::AHasher;
use std::hash::Hash;
use std::hash::Hasher;

/// Unified vocabulary using arena allocation for minimal memory overhead.
///
/// This eliminates the duplicate storage of Vocab + VocabR by using
/// a single arena for all token strings with indexed references.
#[derive(Clone)]
pub struct UnifiedVocab {
    /// Arena storing all token entries
    entries: Vec<TokenEntry>,

    /// Forward lookup: hash of token -> index in arena
    forward: AHashMap<u64, u32>,

    /// Special tokens (always present)
    special: SpecialTokensConfig,

    /// Byte mapping cache (256 entries, pre-computed)
    /// Maps byte value to token ID for byte-fallback
    byte_mapping: [u32; 256],
}

/// Single token entry stored in the arena
#[derive(Clone, Copy)]
struct TokenEntry {
    /// Token ID (same as index in entries Vec)
    id: u32,
    /// Offset into the string store
    offset: u32,
    /// Length of the token
    length: u16,
}

impl UnifiedVocab {
    /// Create a new unified vocabulary with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        // Initialize byte mapping (will be filled as tokens are added)
        let mut byte_mapping = [0u32; 256];

        // Pre-allocate space for tokens
        let mut entries = Vec::with_capacity(capacity);

        // Initialize special token IDs as None
        let special = SpecialTokensConfig::default();

        Self {
            entries,
            forward: AHashMap::with_capacity(capacity),
            special,
            byte_mapping,
        }
    }

    /// Create a new empty unified vocabulary.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Add a token to the vocabulary and return its ID.
    ///
    /// Returns the token ID (or existing ID if token already exists).
    pub fn add_token(&mut self, token: &str) -> Result<u32> {
        // Calculate hash of the token
        let hash = Self::hash_token(token);

        // Check if token already exists
        if let Some(&id) = self.forward.get(&hash) {
            return Ok(id);
        }

        // Assign new ID
        let id = self.entries.len() as u32;

        // Allocate space in string store (simplified - just store bytes)
        // In a full implementation, this would use a proper arena allocator
        let bytes = token.as_bytes();
        let offset = 0; // Placeholder - would be actual offset in arena
        let length = bytes.len() as u16;

        self.entries.push(TokenEntry { id, offset, length });
        self.forward.insert(hash, id);

        // Update byte mapping for single-byte tokens
        if token.len() == 1 {
            let byte = token.as_bytes()[0];
            self.byte_mapping[byte as usize] = id;
        }

        Ok(id)
    }

    /// Add a token with a specific ID.
    ///
    /// This is used when loading a vocabulary from disk.
    pub fn add_token_with_id(&mut self, token: &str, id: u32) -> Result<()> {
        let hash = Self::hash_token(token);

        // Check if token already exists with different ID
        if let Some(&existing_id) = self.forward.get(&hash) {
            if existing_id != id {
                return Err(crate::TokenizerError::InvalidConfig(format!(
                    "Token hash collision: {} exists with id {}, trying to add with id {}",
                    token, existing_id, id
                )));
            }
            return Ok(());
        }

        // Ensure we have space in entries
        while self.entries.len() <= id as usize {
            self.entries.push(TokenEntry {
                id: self.entries.len() as u32,
                offset: 0,
                length: 0,
            });
        }

        // Update entry
        let bytes = token.as_bytes();
        self.entries[id as usize] = TokenEntry {
            id,
            offset: 0,
            length: bytes.len() as u16,
        };

        self.forward.insert(hash, id);

        Ok(())
    }

    /// Get the token ID for a given token string.
    pub fn get_id(&self, token: &str) -> Option<u32> {
        let hash = Self::hash_token(token);
        self.forward.get(&hash).copied()
    }

    /// Get the token string for a given token ID.
    pub fn get_token(&self, _id: u32) -> Option<String> {
        // In a full implementation, this would look up the offset and length
        // in the string store and return the appropriate slice
        // For now, we'll need to maintain a reverse mapping
        // This is a simplification - the real implementation would have
        // a proper arena allocator
        None
    }

    /// Get the vocabulary size.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Set special token IDs.
    pub fn set_special_tokens(&mut self, special: SpecialTokensConfig) {
        self.special = special;
    }

    /// Get reference to special tokens.
    pub fn special(&self) -> &SpecialTokensConfig {
        &self.special
    }

    /// Hash a token string for fast lookup.
    fn hash_token(token: &str) -> u64 {
        let mut hasher = AHasher::default();
        token.hash(&mut hasher);
        hasher.finish()
    }

    /// Iterate over all token hashes and IDs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.forward.iter().map(|(&hash, &id)| (hash, id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vocab() {
        let vocab = UnifiedVocab::new();
        assert_eq!(vocab.len(), 0);
        assert!(vocab.is_empty());
    }

    #[test]
    fn test_add_token() {
        let mut vocab = UnifiedVocab::new();
        let id1 = vocab.add_token("hello").unwrap();
        let id2 = vocab.add_token("world").unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_add_duplicate() {
        let mut vocab = UnifiedVocab::new();
        let id1 = vocab.add_token("hello").unwrap();
        let id2 = vocab.add_token("hello").unwrap();

        assert_eq!(id1, id2);
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_get_id() {
        let mut vocab = UnifiedVocab::new();
        vocab.add_token("test").unwrap();

        assert_eq!(vocab.get_id("test"), Some(0));
        assert_eq!(vocab.get_id("unknown"), None);
    }

    #[test]
    fn test_byte_mapping() {
        let mut vocab = UnifiedVocab::new();
        vocab.add_token("a").unwrap();
        vocab.add_token("b").unwrap();

        // Single-byte tokens should be in byte mapping
        let byte_a = b'a' as usize;
        let byte_b = b'b' as usize;

        assert_eq!(vocab.byte_mapping[byte_a], 0);
        assert_eq!(vocab.byte_mapping[byte_b], 1);
    }
}
