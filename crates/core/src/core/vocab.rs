//! Vocabulary storage and lookup.
//!
//! This module provides efficient vocabulary storage using AHashMap for fast lookups
//! and CompactString for memory-efficient string storage.

use crate::error::{Result, TokenizerError};
use ahash::AHashMap;
use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Forward mapping: token string -> ID
pub type Vocab = AHashMap<CompactString, u32>;

/// Reverse mapping: ID -> token string
pub type VocabR = AHashMap<u32, CompactString>;

/// Vocabulary with forward and reverse mappings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Forward mapping: token string -> ID
    pub vocab: Vocab,
    /// Reverse mapping: ID -> token string
    pub vocab_r: VocabR,
    /// Special token IDs (cached for fast access)
    pub special: SpecialTokens,
}

impl Vocabulary {
    /// Create a new empty vocabulary.
    pub fn new() -> Self {
        Self {
            vocab: Vocab::new(),
            vocab_r: VocabR::new(),
            special: SpecialTokens::default(),
        }
    }

    /// Create a new vocabulary with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vocab: Vocab::with_capacity(capacity),
            vocab_r: VocabR::with_capacity(capacity),
            special: SpecialTokens::default(),
        }
    }

    /// Add a token to the vocabulary.
    ///
    /// Returns the ID assigned to the token.
    pub fn add_token(&mut self, token: &str) -> Result<u32> {
        let token = CompactString::new(token);

        // Check if token already exists
        if let Some(&id) = self.vocab.get(&token) {
            return Ok(id);
        }

        let id = self.vocab.len() as u32;
        self.vocab_r.insert(id, token.clone());
        self.vocab.insert(token, id);

        Ok(id)
    }

    /// Add a token with a specific ID.
    ///
    /// Returns an error if the ID is already taken.
    pub fn add_token_with_id(&mut self, token: &str, id: u32) -> Result<()> {
        let token = CompactString::new(token);

        if self.vocab_r.contains_key(&id) {
            return Err(TokenizerError::InvalidConfig(format!(
                "Token ID {} already exists",
                id
            )));
        }

        self.vocab_r.insert(id, token.clone());
        self.vocab.insert(token, id);

        Ok(())
    }

    /// Get the ID for a token string.
    #[inline]
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get the token string for an ID.
    #[inline]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(|s| s.as_str())
    }

    /// Get the size of the vocabulary.
    #[inline]
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Check if the vocabulary is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Add special tokens to the vocabulary.
    pub fn add_special_tokens(&mut self, special: SpecialTokensConfig) -> Result<()> {
        if let Some(pad) = special.pad {
            self.special.pad = Some(self.add_token(&pad)?);
        }
        if let Some(unk) = special.unk {
            self.special.unk = Some(self.add_token(&unk)?);
        }
        if let Some(bos) = special.bos {
            self.special.bos = Some(self.add_token(&bos)?);
        }
        if let Some(eos) = special.eos {
            self.special.eos = Some(self.add_token(&eos)?);
        }
        if let Some(mask) = special.mask {
            self.special.mask = Some(self.add_token(&mask)?);
        }
        if let Some(user) = special.user {
            self.special.user = Some(self.add_token(&user)?);
        }
        if let Some(assistant) = special.assistant {
            self.special.assistant = Some(self.add_token(&assistant)?);
        }
        if let Some(system) = special.system {
            self.special.system = Some(self.add_token(&system)?);
        }

        Ok(())
    }

    /// Get Arc-wrapped clones of the vocabulary HashMaps for efficient sharing.
    ///
    /// This method creates Arc-wrapped clones of the internal HashMaps,
    /// allowing them to be shared with encoders without deep cloning.
    /// The Arc cloning is cheap (just incrementing reference count).
    pub fn get_arcs(&self) -> (Arc<Vocab>, Arc<VocabR>) {
        (Arc::new(self.vocab.clone()), Arc::new(self.vocab_r.clone()))
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Special token IDs cached for fast access.
///
/// These are used extensively during tokenization, so we cache the IDs
/// to avoid repeated lookups.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Padding token ID
    pub pad: Option<u32>,
    /// Unknown token ID
    pub unk: Option<u32>,
    /// Beginning of sequence token ID
    pub bos: Option<u32>,
    /// End of sequence token ID
    pub eos: Option<u32>,
    /// Mask token ID
    pub mask: Option<u32>,
    /// User role token ID (for chat)
    pub user: Option<u32>,
    /// Assistant role token ID (for chat)
    pub assistant: Option<u32>,
    /// System role token ID (for chat)
    pub system: Option<u32>,
}

impl SpecialTokens {
    /// Check if an ID is a special token.
    #[inline]
    pub fn is_special(&self, id: u32) -> bool {
        Some(id) == self.pad
            || Some(id) == self.unk
            || Some(id) == self.bos
            || Some(id) == self.eos
            || Some(id) == self.mask
            || Some(id) == self.user
            || Some(id) == self.assistant
            || Some(id) == self.system
    }
}

/// Configuration for special tokens.
///
/// Used during tokenizer construction to specify which special tokens to add.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokensConfig {
    pub pad: Option<String>,
    pub unk: Option<String>,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub mask: Option<String>,
    pub user: Option<String>,
    pub assistant: Option<String>,
    pub system: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_token() {
        let mut vocab = Vocabulary::new();
        let id1 = vocab.add_token("hello").unwrap();
        let id2 = vocab.add_token("world").unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(vocab.get_id("hello"), Some(0));
        assert_eq!(vocab.get_id("world"), Some(1));
        assert_eq!(vocab.get_token(0), Some("hello"));
        assert_eq!(vocab.get_token(1), Some("world"));
    }

    #[test]
    fn test_add_duplicate_token() {
        let mut vocab = Vocabulary::new();
        let id1 = vocab.add_token("hello").unwrap();
        let id2 = vocab.add_token("hello").unwrap();

        assert_eq!(id1, id2);
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_add_token_with_id() {
        let mut vocab = Vocabulary::new();
        vocab.add_token_with_id("hello", 5).unwrap();
        vocab.add_token_with_id("world", 10).unwrap();

        assert_eq!(vocab.get_id("hello"), Some(5));
        assert_eq!(vocab.get_id("world"), Some(10));
        assert_eq!(vocab.get_token(5), Some("hello"));
        assert_eq!(vocab.get_token(10), Some("world"));
    }

    #[test]
    fn test_special_tokens() {
        let mut vocab = Vocabulary::new();
        vocab
            .add_special_tokens(SpecialTokensConfig {
                bos: Some("<bos>".to_string()),
                eos: Some("<eos>".to_string()),
                unk: Some("<unk>".to_string()),
                ..Default::default()
            })
            .unwrap();

        assert!(vocab.special.bos.is_some());
        assert!(vocab.special.eos.is_some());
        assert!(vocab.special.unk.is_some());
        assert!(vocab.special.is_special(vocab.special.bos.unwrap()));
        assert!(vocab.special.is_special(vocab.special.eos.unwrap()));
        assert!(vocab.special.is_special(vocab.special.unk.unwrap()));
    }
}
