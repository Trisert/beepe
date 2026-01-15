//! Save functionality for trained tokenizers.
//!
//! This module provides methods for saving trained tokenizers to disk
//! in various formats.

use super::super::EncodingMode;
use super::format::{
    SerializedConfig, SerializedMerge, SerializedSpecialTokens, SerializedTokenizer,
};
use ahash::AHashMap;
use beepe_core::{Pair, Result, TokenizerError, Vocabulary};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Tokenizer saver - handles saving trained models.
pub struct TokenizerSaver<'a> {
    /// Vocabulary reference
    vocab: &'a Vocabulary,
    /// Merge rules reference
    merges: &'a AHashMap<Pair, (u32, u32)>,
    /// Encoding mode
    encoding_mode: EncodingMode,
}

impl<'a> TokenizerSaver<'a> {
    /// Create a new tokenizer saver.
    pub fn new(
        vocab: &'a Vocabulary,
        merges: &'a AHashMap<Pair, (u32, u32)>,
        encoding_mode: EncodingMode,
    ) -> Self {
        Self {
            vocab,
            merges,
            encoding_mode,
        }
    }

    /// Save the tokenizer to a directory in custom JSON format.
    ///
    /// This saves a single `tokenizer.json` file containing all model data.
    ///
    /// # Arguments
    /// * `path` - Directory path to save to
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path).map_err(|e| {
            TokenizerError::Save(format!(
                "Failed to create directory {}: {}",
                path.display(),
                e
            ))
        })?;

        let file_path = path.join("tokenizer.json");
        let file = File::create(&file_path).map_err(|e| {
            TokenizerError::Save(format!(
                "Failed to create file {}: {}",
                file_path.display(),
                e
            ))
        })?;

        let writer = BufWriter::new(file);
        let serialized = self.serialize();
        serde_json::to_writer_pretty(writer, &serialized)
            .map_err(|e| TokenizerError::Save(format!("Failed to serialize tokenizer: {}", e)))?;

        Ok(())
    }

    /// Save in HuggingFace format (vocab.json + merges.txt).
    ///
    /// This creates two files:
    /// - `vocab.json`: Token to ID mapping
    /// - `merges.txt`: Merge rules, one per line
    pub fn save_huggingface(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path).map_err(|e| {
            TokenizerError::Save(format!(
                "Failed to create directory {}: {}",
                path.display(),
                e
            ))
        })?;

        // Save vocab.json
        let vocab_path = path.join("vocab.json");
        let vocab_file = File::create(&vocab_path)
            .map_err(|e| TokenizerError::Save(format!("Failed to create vocab.json: {}", e)))?;
        let vocab_writer = BufWriter::new(vocab_file);

        let vocab_map: std::collections::HashMap<String, u32> = self
            .vocab
            .vocab
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        serde_json::to_writer_pretty(vocab_writer, &vocab_map)
            .map_err(|e| TokenizerError::Save(format!("Failed to serialize vocab: {}", e)))?;

        // Save merges.txt
        let merges_path = path.join("merges.txt");
        let mut merges_file = std::fs::File::create(&merges_path)
            .map_err(|e| TokenizerError::Save(format!("Failed to create merges.txt: {}", e)))?;

        use std::io::Write;

        // Sort merges by rank
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|(_, &(rank, _))| rank);

        for (pair, (_rank, _new_id)) in sorted_merges {
            let token1 = self.vocab.get_token(pair.0).unwrap_or("");
            let token2 = self.vocab.get_token(pair.1).unwrap_or("");
            writeln!(merges_file, "{} {}", token1, token2)
                .map_err(|e| TokenizerError::Save(format!("Failed to write merges: {}", e)))?;
        }

        Ok(())
    }

    /// Serialize the tokenizer to a structure.
    fn serialize(&self) -> SerializedTokenizer {
        // Convert vocab
        let vocab: std::collections::HashMap<String, u32> = self
            .vocab
            .vocab
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        let vocab_r: std::collections::HashMap<u32, String> = self
            .vocab
            .vocab_r
            .iter()
            .map(|(k, v)| (*k, v.to_string()))
            .collect();

        // Convert merges
        let merges: Vec<SerializedMerge> = self
            .merges
            .iter()
            .map(|(pair, &(rank, new_id))| SerializedMerge {
                pair: (
                    self.vocab.get_token(pair.0).unwrap_or("").to_string(),
                    self.vocab.get_token(pair.1).unwrap_or("").to_string(),
                ),
                rank,
                new_token_id: new_id,
            })
            .collect();

        // Serialize special tokens
        let special_tokens = SerializedSpecialTokens {
            pad: self
                .vocab
                .special
                .pad
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            unk: self
                .vocab
                .special
                .unk
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            bos: self
                .vocab
                .special
                .bos
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            eos: self
                .vocab
                .special
                .eos
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            mask: self
                .vocab
                .special
                .mask
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            user: self
                .vocab
                .special
                .user
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            assistant: self
                .vocab
                .special
                .assistant
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
            system: self
                .vocab
                .special
                .system
                .and_then(|id| self.vocab.get_token(id).map(|s| s.to_string())),
        };

        SerializedTokenizer {
            version: env!("CARGO_PKG_VERSION").to_string(),
            vocab,
            vocab_r,
            merges,
            special_tokens,
            config: SerializedConfig {
                vocab_size: self.vocab.len(),
                min_frequency: 2,
                encoding_mode: format!("{:?}", self.encoding_mode),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beepe_core::Vocabulary;

    #[test]
    fn test_serialize() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello").unwrap();
        vocab.add_token("world").unwrap();

        let merges: AHashMap<Pair, (u32, u32)> = AHashMap::new();

        let saver = TokenizerSaver::new(&vocab, &merges, EncodingMode::ByteLevel);
        let serialized = saver.serialize();

        assert_eq!(serialized.vocab.len(), 2);
        assert_eq!(serialized.version, env!("CARGO_PKG_VERSION"));
    }
}
