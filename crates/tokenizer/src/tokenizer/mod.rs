//! Main tokenizer implementation.
//!
//! This module provides the high-level `Tokenizer` struct that integrates
//! vocabulary, merge rules, and encoding modes.

use crate::pre_tokenizer::{Normalizer, Splitter};
use beepe_core::{
    ByteLevelEncoderV2, CharLevelEncoder, Result, SpecialTokensConfig, TokenizerError, Vocabulary,
};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

// Type alias for convenience
use ahash::AHashMap as AHashmap;

/// Encoding mode for the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMode {
    /// Byte-level encoding (tiktoken-style)
    ByteLevel,
    /// Character/grapheme-level encoding
    CharLevel,
    /// Hybrid mode
    Hybrid,
}

/// Configuration for building a tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for merges during training
    pub min_frequency: u64,
    /// Encoding mode
    pub encoding_mode: EncodingMode,
    /// Special tokens configuration
    pub special_tokens: SpecialTokensConfig,
    /// Capacity for encoding cache
    pub cache_capacity: usize,
    /// Use entropy-weighted BPE training (better compression, slightly slower)
    pub use_entropy_weighted_training: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30_000,
            min_frequency: 2,
            encoding_mode: EncodingMode::ByteLevel,
            special_tokens: SpecialTokensConfig::default(),
            cache_capacity: 1000,
            use_entropy_weighted_training: true, // Enable by default for better compression
        }
    }
}

/// Builder for creating a tokenizer.
#[derive(Clone)]
pub struct TokenizerBuilder {
    config: TokenizerConfig,
}

impl Default for TokenizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerBuilder {
    /// Create a new tokenizer builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
        }
    }

    /// Set the target vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set the minimum frequency for merges.
    pub fn min_frequency(mut self, freq: u64) -> Self {
        self.config.min_frequency = freq;
        self
    }

    /// Set the encoding mode.
    pub fn encoding_mode(mut self, mode: EncodingMode) -> Self {
        self.config.encoding_mode = mode;
        self
    }

    /// Set special tokens.
    pub fn with_special_tokens(mut self, tokens: SpecialTokensConfig) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Build the tokenizer.
    pub fn build(self) -> Result<Tokenizer> {
        Tokenizer::new(self.config)
    }
}

/// Main tokenizer struct.
///
/// This is the high-level API that integrates vocabulary, merge rules,
/// and encoding modes into a single interface.
pub struct Tokenizer {
    /// Vocabulary
    vocab: Vocabulary,
    /// Merge rules
    merges: AHashmap<(u32, u32), (u32, u32)>,
    /// Configuration
    config: TokenizerConfig,
    /// Byte-level encoder (V2 with Arc sharing)
    byte_encoder: Option<Arc<ByteLevelEncoderV2>>,
    /// Character-level encoder
    char_encoder: Option<CharLevelEncoder>,
    /// Text splitter
    splitter: Splitter,
    /// Unicode normalizer
    normalizer: Normalizer,
}

impl Tokenizer {
    /// Create a new tokenizer with the given configuration.
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let mut vocab = Vocabulary::with_capacity(config.vocab_size);

        // Add special tokens
        vocab.add_special_tokens(config.special_tokens.clone())?;

        // Initialize byte vocabulary for byte-level encoding
        Self::init_byte_vocab(&mut vocab)?;

        // Get Arc-wrapped vocabularies for zero-copy sharing with encoders
        let (vocab_arc, vocab_r_arc) = vocab.get_arcs();
        let merges_arc = Arc::new(AHashmap::new());

        // Create the byte-level encoder V2 with Arc sharing (zero-copy)
        let byte_encoder = Some(ByteLevelEncoderV2::with_arcs(
            vocab_arc.clone(),
            vocab_r_arc.clone(),
            merges_arc.clone(),
        ));

        // Create character-level encoder (still uses old approach for now)
        // TODO: Migrate CharLevelEncoder to V2 as well
        let vocab_map = (*vocab_arc).clone();
        let vocab_r_map = (*vocab_r_arc).clone();
        let char_encoder = Some(CharLevelEncoder::new(
            vocab_map,
            vocab_r_map,
            AHashmap::new(),
        ));

        Ok(Self {
            vocab,
            merges: AHashmap::new(),
            config,
            byte_encoder,
            char_encoder,
            splitter: Splitter::default(),
            normalizer: Normalizer::default(),
        })
    }

    /// Initialize byte vocabulary for byte-level encoding.
    fn init_byte_vocab(vocab: &mut Vocabulary) -> Result<()> {
        // Add bytes 0-255 as unicode characters (tiktoken's approach)
        for i in 0u32..256 {
            let codepoint = 256 + i;
            let ch = char::from_u32(codepoint).ok_or_else(|| {
                TokenizerError::InvalidConfig(format!(
                    "Invalid codepoint {} for byte {}",
                    codepoint, i
                ))
            })?;
            vocab.add_token(&ch.to_string())?;
        }

        Ok(())
    }

    /// Convert vocabulary entries to byte-mapped form for ByteLevel encoding.
    ///
    /// This ensures that when the encoder searches for byte-mapped sequences,
    /// it finds the learned tokens from training.
    fn convert_vocab_to_byte_mapped(original_vocab: Vocabulary) -> Result<Vocabulary> {
        let mut new_vocab = Vocabulary::with_capacity(original_vocab.len());

        // Copy special tokens as-is
        new_vocab.special = original_vocab.special.clone();

        // Convert each vocabulary entry to byte-mapped form
        for (token_str, &token_id) in original_vocab.vocab.iter() {
            // Convert the token string to byte-mapped characters
            let byte_mapped: String = token_str
                .as_bytes()
                .iter()
                .map(|&b| char::from_u32(256 + b as u32).unwrap())
                .collect();

            // Add with the original ID
            new_vocab.add_token_with_id(&byte_mapped, token_id)?;
        }

        Ok(new_vocab)
    }

    /// Create a tokenizer builder.
    pub fn builder() -> TokenizerBuilder {
        TokenizerBuilder::new()
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_special_tokens` - Whether to add special tokens (BOS, EOS, etc.)
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        // Check text size limit (1M characters like tiktoken)
        if text.len() > 1_000_000 {
            return Err(TokenizerError::Tokenization(format!(
                "Text too large: {} characters (max: 1,000,000)",
                text.len()
            )));
        }

        // Apply normalization
        let normalized_text = self.normalizer.normalize(text);

        // Apply pre-tokenization splitting
        let splits = self.splitter.split(&normalized_text);

        let mut all_ids = Vec::new();

        // Encode each split
        for split in splits {
            let ids = match self.config.encoding_mode {
                EncodingMode::ByteLevel => {
                    let encoder = self.byte_encoder.as_ref().ok_or_else(|| {
                        TokenizerError::Tokenization("Byte encoder not initialized".to_string())
                    })?;
                    encoder.encode(&split)?
                }
                EncodingMode::CharLevel => {
                    let encoder = self.char_encoder.as_ref().ok_or_else(|| {
                        TokenizerError::Tokenization("Char encoder not initialized".to_string())
                    })?;
                    encoder.encode(&split)?
                }
                EncodingMode::Hybrid => {
                    // Try byte-level first, fall back to char-level
                    let byte_encoder = self.byte_encoder.as_ref().ok_or_else(|| {
                        TokenizerError::Tokenization("Byte encoder not initialized".to_string())
                    })?;
                    match byte_encoder.encode(&split) {
                        Ok(ids) => ids,
                        Err(_) => {
                            let char_encoder = self.char_encoder.as_ref().ok_or_else(|| {
                                TokenizerError::Tokenization(
                                    "Char encoder not initialized".to_string(),
                                )
                            })?;
                            char_encoder.encode(&split)?
                        }
                    }
                }
            };
            all_ids.extend(ids);
        }

        let mut ids = all_ids;

        // Add special tokens if requested
        if add_special_tokens {
            if let Some(bos) = self.vocab.special.bos {
                ids.insert(0, bos);
            }
            if let Some(eos) = self.vocab.special.eos {
                ids.push(eos);
            }
        }

        Ok(Encoding {
            ids,
            text: text.to_string(),
        })
    }

    /// Encode a batch of texts (parallelized).
    pub fn encode_batch(
        &self,
        texts: &[String],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        use rayon::prelude::*;

        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect::<std::result::Result<Vec<_>, _>>()
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    /// * `ids` - The token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens during decoding
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let decoder = self.byte_encoder.as_ref().unwrap();

        // Check for invalid token IDs
        for &id in ids {
            if id as usize >= self.vocab.vocab_r.len() {
                return format!("<invalid token {}>", id);
            }
        }

        let filtered_ids: Vec<u32> = if skip_special_tokens {
            ids.iter()
                .filter(|&&id| !self.vocab.special.is_special(id))
                .copied()
                .collect()
        } else {
            ids.to_vec()
        };

        decoder.decode(&filtered_ids).unwrap_or_else(|_| {
            // If decoding fails, return a placeholder
            "<decoding error>".to_string()
        })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get a reference to the vocabulary.
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Train the tokenizer on text data.
    ///
    /// # Arguments
    /// * `data` - Training text data
    pub fn train(&mut self, data: &str) -> Result<()> {
        use beepe_training::{BpeTrainerV2, TrainingConfigV2};

        // Save special token IDs and strings before training
        let saved_special = self.vocab.special.clone();
        let special_tokens_config = self.config.special_tokens.clone();

        // Use entropy-weighted trainer for better compression
        let training_config = TrainingConfigV2 {
            vocab_size: self.config.vocab_size,
            min_frequency: self.config.min_frequency,
            parallel: true,
            ..Default::default()
        };

        let mut trainer = BpeTrainerV2::new(training_config);
        let (vocab, merges) = trainer.train(data)?;

        // For ByteLevel encoding, convert vocabulary entries to byte-mapped form
        // This ensures the encoder (which uses byte-mapped characters) can find the learned tokens
        if self.config.encoding_mode == EncodingMode::ByteLevel {
            self.vocab = Self::convert_vocab_to_byte_mapped(vocab)?;
        } else {
            self.vocab = vocab;
        }
        self.merges = merges;

        // Re-initialize byte vocabulary after training
        // (training creates a new vocab, losing byte fallback characters)
        Self::init_byte_vocab(&mut self.vocab)?;

        // Restore special tokens by adding them back to the vocabulary
        // This ensures they have consistent IDs and are accessible
        self.vocab.add_special_tokens(special_tokens_config)?;

        // Restore special token IDs from saved state
        self.vocab.special = saved_special;

        // Rebuild byte encoder with new vocab/merges using Arc sharing
        let (vocab_arc, vocab_r_arc) = self.vocab.get_arcs();
        let merges_arc = Arc::new(self.merges.clone());
        self.byte_encoder = Some(ByteLevelEncoderV2::with_arcs(
            vocab_arc,
            vocab_r_arc,
            merges_arc,
        ));

        Ok(())
    }

    /// Save the tokenizer to a directory.
    ///
    /// # Arguments
    /// * `path` - Directory path to save to
    pub fn save(&self, path: &Path) -> Result<()> {
        use crate::io::save::TokenizerSaver;

        let saver = TokenizerSaver::new(&self.vocab, &self.merges, self.config.encoding_mode);
        saver.save(path)
    }

    /// Load a tokenizer from a directory.
    ///
    /// # Arguments
    /// * `path` - Directory path to load from
    pub fn load(path: &Path) -> Result<Self> {
        use crate::io::{format::SerializedTokenizer, load::TokenizerLoader};

        // Load the config from JSON first to get encoding mode
        let file_path = path.join("tokenizer.json");
        let file = File::open(&file_path)
            .map_err(|e| TokenizerError::Load(format!("Failed to open tokenizer.json: {}", e)))?;
        let reader = BufReader::new(file);
        let serialized: SerializedTokenizer = serde_json::from_reader(reader)
            .map_err(|e| TokenizerError::Load(format!("Failed to deserialize tokenizer: {}", e)))?;

        // Parse encoding mode from config
        let encoding_mode = match serialized.config.encoding_mode.as_str() {
            "ByteLevel" => EncodingMode::ByteLevel,
            "CharLevel" => EncodingMode::CharLevel,
            "Hybrid" => EncodingMode::Hybrid,
            _ => EncodingMode::ByteLevel, // Default fallback
        };

        let (vocab, merges) = TokenizerLoader::load(path)?;

        // Create Arc-wrapped vocabularies for zero-copy sharing
        let (vocab_arc, vocab_r_arc) = vocab.get_arcs();
        let merges_arc = Arc::new(merges.clone());

        let byte_encoder = Some(ByteLevelEncoderV2::with_arcs(
            vocab_arc.clone(),
            vocab_r_arc.clone(),
            merges_arc.clone(),
        ));

        // Character-level encoder still uses old approach
        let vocab_map = (*vocab_arc).clone();
        let vocab_r_map = (*vocab_r_arc).clone();
        let char_encoder = Some(CharLevelEncoder::new(
            vocab_map,
            vocab_r_map,
            merges.clone(),
        ));

        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            min_frequency: serialized.config.min_frequency,
            encoding_mode,
            special_tokens: SpecialTokensConfig::default(),
            cache_capacity: 1000,
            use_entropy_weighted_training: true, // Default for loaded models
        };

        Ok(Self {
            vocab,
            merges,
            config,
            byte_encoder,
            char_encoder,
            splitter: Splitter::default(),
            normalizer: Normalizer::default(),
        })
    }

    /// Load a tokenizer from HuggingFace format.
    ///
    /// # Arguments
    /// * `path` - Directory path containing vocab.json and merges.txt
    pub fn load_huggingface(path: &Path) -> Result<Self> {
        use crate::io::load::TokenizerLoader;

        let (vocab, merges) = TokenizerLoader::load_huggingface(path)?;

        // Create Arc-wrapped vocabularies for zero-copy sharing
        let (vocab_arc, vocab_r_arc) = vocab.get_arcs();
        let merges_arc = Arc::new(merges.clone());

        let byte_encoder = Some(ByteLevelEncoderV2::with_arcs(
            vocab_arc.clone(),
            vocab_r_arc.clone(),
            merges_arc.clone(),
        ));

        // Character-level encoder still uses old approach
        let vocab_map = (*vocab_arc).clone();
        let vocab_r_map = (*vocab_r_arc).clone();
        let char_encoder = Some(CharLevelEncoder::new(
            vocab_map,
            vocab_r_map,
            merges.clone(),
        ));

        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            min_frequency: 2,
            encoding_mode: EncodingMode::ByteLevel,
            special_tokens: SpecialTokensConfig::default(),
            cache_capacity: 1000,
            use_entropy_weighted_training: true, // Default for loaded models
        };

        Ok(Self {
            vocab,
            merges,
            config,
            byte_encoder,
            char_encoder,
            splitter: Splitter::default(),
            normalizer: Normalizer::default(),
        })
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = TokenizerError;

    fn from_str(_s: &str) -> std::result::Result<Self, Self::Err> {
        // TODO: Implement deserialization
        Err(TokenizerError::Tokenization(
            "Deserialization not yet implemented".to_string(),
        ))
    }
}

/// Result of encoding text.
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token IDs
    pub ids: Vec<u32>,
    /// Original text
    pub text: String,
}

impl Encoding {
    /// Get the number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if the encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Get the tokens as strings.
    pub fn get_tokens(&self) -> Vec<String> {
        self.ids.iter().map(|id| id.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let tokenizer = Tokenizer::builder()
            .vocab_size(1000)
            .min_frequency(5)
            .encoding_mode(EncodingMode::ByteLevel)
            .build();

        assert!(tokenizer.is_ok());
        let tokenizer = tokenizer.unwrap();
        assert_eq!(tokenizer.vocab_size(), 256); // 256 bytes from init
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = Tokenizer::builder()
            .encoding_mode(EncodingMode::ByteLevel)
            .build()
            .unwrap();

        let text = "Hello, world!";
        let encoding = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&encoding.ids, false);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let mut tokenizer = Tokenizer::builder()
            .encoding_mode(EncodingMode::ByteLevel)
            .with_special_tokens(SpecialTokensConfig {
                bos: Some("<bos>".to_string()),
                eos: Some("<eos>".to_string()),
                ..Default::default()
            })
            .build()
            .unwrap();

        // Create encoder manually for special token support
        let (vocab_arc, vocab_r_arc) = tokenizer.vocab.get_arcs();
        let merges_arc = Arc::new(AHashmap::new());
        let encoder = ByteLevelEncoderV2::with_arcs(vocab_arc, vocab_r_arc, merges_arc);
        tokenizer.byte_encoder = Some(encoder);

        let text = "Hello";
        let encoding = tokenizer.encode(text, true).unwrap();

        // Should have BOS + content + EOS
        assert!(encoding.ids.len() >= 3);
    }

    #[test]
    fn test_encoding_length() {
        let tokenizer = Tokenizer::builder().build().unwrap();

        let encoding = tokenizer.encode("Hello, world!", false).unwrap();
        assert!(!encoding.is_empty());
        assert_eq!(encoding.len(), encoding.ids.len());
    }
}
