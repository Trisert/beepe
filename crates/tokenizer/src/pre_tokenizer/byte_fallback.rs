//! Byte-level fallback for unknown tokens.
//!
//! This module provides fallback encoding for tokens that are not
//! in the vocabulary, falling back to byte-level representation.

use beepe_core::Vocabulary;

/// Byte-level fallback encoder.
pub struct ByteFallback {
    /// Whether fallback is enabled
    enabled: bool,
}

impl ByteFallback {
    /// Create a new byte fallback handler.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Enable byte fallback.
    pub fn enabled() -> Self {
        Self::new(true)
    }

    /// Disable byte fallback.
    pub fn disabled() -> Self {
        Self::new(false)
    }

    /// Encode an unknown token using byte-level fallback.
    ///
    /// Returns a vector of token IDs representing the bytes of the unknown token.
    pub fn encode_unknown(&self, text: &str, vocab: &Vocabulary) -> Vec<u32> {
        if !self.enabled {
            return vec![vocab.special.unk.unwrap_or(0)];
        }

        // Encode each byte as a separate token
        text.bytes()
            .map(|b| {
                // Try to get the byte token from vocab
                let byte_str = format!("<0x{:02x}>", b);
                vocab.get_id(&byte_str).unwrap_or_else(|| {
                    // Fallback to UNK token
                    vocab.special.unk.unwrap_or(0)
                })
            })
            .collect()
    }

    /// Check if fallback is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for ByteFallback {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beepe_core::SpecialTokensConfig;

    #[test]
    fn test_byte_fallback_disabled() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello").unwrap();
        vocab
            .add_special_tokens(SpecialTokensConfig {
                unk: Some("<unk>".to_string()),
                ..Default::default()
            })
            .unwrap();

        let fallback = ByteFallback::disabled();
        let result = fallback.encode_unknown("xyz", &vocab);

        // Should return UNK token
        assert_eq!(result, vec![vocab.special.unk.unwrap()]);
    }

    #[test]
    fn test_byte_fallback_enabled() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("<0x61>").unwrap(); // 'a' as byte
        vocab
            .add_special_tokens(SpecialTokensConfig {
                unk: Some("<unk>".to_string()),
                ..Default::default()
            })
            .unwrap();

        let fallback = ByteFallback::enabled();
        let result = fallback.encode_unknown("a", &vocab);

        // Should return the byte token
        assert_eq!(result, vec![vocab.get_id("<0x61>").unwrap()]);
    }

    #[test]
    fn test_is_enabled() {
        let enabled = ByteFallback::enabled();
        assert!(enabled.is_enabled());

        let disabled = ByteFallback::disabled();
        assert!(!disabled.is_enabled());
    }
}
