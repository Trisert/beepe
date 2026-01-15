//! Load functionality for pre-trained tokenizers.
//!
//! This module provides methods for loading tokenizers from disk
//! in various formats.

use super::format::SerializedTokenizer;
use ahash::AHashMap;
use beepe_core::{Pair, Result, TokenizerError, Vocabulary};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Tokenizer loader - handles loading trained models.
pub struct TokenizerLoader;

impl TokenizerLoader {
    /// Load a tokenizer from a directory in custom JSON format.
    ///
    /// Expects a `tokenizer.json` file in the given directory.
    ///
    /// # Arguments
    /// * `path` - Directory path to load from
    pub fn load(path: &Path) -> Result<(Vocabulary, AHashMap<Pair, (u32, u32)>)> {
        let file_path = path.join("tokenizer.json");
        let file = File::open(&file_path).map_err(|e| {
            TokenizerError::Load(format!(
                "Failed to open file {}: {}",
                file_path.display(),
                e
            ))
        })?;

        let reader = BufReader::new(file);
        let serialized: SerializedTokenizer = serde_json::from_reader(reader)
            .map_err(|e| TokenizerError::Load(format!("Failed to deserialize tokenizer: {}", e)))?;

        Self::deserialize(serialized)
    }

    /// Load from HuggingFace format (vocab.json + merges.txt).
    ///
    /// Expects two files in the given directory:
    /// - `vocab.json`: Token to ID mapping
    /// - `merges.txt`: Merge rules, one per line
    pub fn load_huggingface(path: &Path) -> Result<(Vocabulary, AHashMap<Pair, (u32, u32)>)> {
        // Load vocab.json
        let vocab_path = path.join("vocab.json");
        let vocab_file = File::open(&vocab_path)
            .map_err(|e| TokenizerError::Load(format!("Failed to open vocab.json: {}", e)))?;
        let vocab_reader = BufReader::new(vocab_file);
        let vocab_map: std::collections::HashMap<String, u32> =
            serde_json::from_reader(vocab_reader)
                .map_err(|e| TokenizerError::Load(format!("Failed to deserialize vocab: {}", e)))?;

        // Load merges.txt
        let merges_path = path.join("merges.txt");
        let merges_content = std::fs::read_to_string(&merges_path)
            .map_err(|e| TokenizerError::Load(format!("Failed to read merges.txt: {}", e)))?;

        // Build vocabulary
        let mut vocab = Vocabulary::with_capacity(vocab_map.len());
        for (token, id) in vocab_map {
            vocab.add_token_with_id(&token, id)?;
        }

        // Parse merges
        let mut merges = AHashMap::new();
        for (line_num, line) in merges_content.lines().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(TokenizerError::Load(format!(
                    "Invalid merge format at line {}: '{}'",
                    line_num + 1,
                    line
                )));
            }

            let token1_id = vocab.get_id(parts[0]).ok_or_else(|| {
                TokenizerError::Load(format!("Unknown token in merges: {}", parts[0]))
            })?;
            let token2_id = vocab.get_id(parts[1]).ok_or_else(|| {
                TokenizerError::Load(format!("Unknown token in merges: {}", parts[1]))
            })?;

            let pair = (token1_id, token2_id);
            let rank = line_num as u32;
            let new_token_id = vocab.len() as u32 + rank; // Approximate

            merges.insert(pair, (rank, new_token_id));
        }

        Ok((vocab, merges))
    }

    /// Deserialize from a serialized structure.
    fn deserialize(data: SerializedTokenizer) -> Result<(Vocabulary, AHashMap<Pair, (u32, u32)>)> {
        let mut vocab = Vocabulary::with_capacity(data.vocab.len());

        // Build vocabulary from serialized data
        for (token, id) in data.vocab {
            vocab.add_token_with_id(&token, id)?;
        }

        // Ensure vocab_r is consistent
        for (id, token) in data.vocab_r {
            if vocab.get_token(id).is_none() {
                vocab.add_token_with_id(&token, id)?;
            }
        }

        // Deserialize merges
        let mut merges = AHashMap::new();
        for merge_data in data.merges {
            let token1_id = vocab.get_id(&merge_data.pair.0).ok_or_else(|| {
                TokenizerError::Load(format!("Unknown token: {}", merge_data.pair.0))
            })?;
            let token2_id = vocab.get_id(&merge_data.pair.1).ok_or_else(|| {
                TokenizerError::Load(format!("Unknown token: {}", merge_data.pair.1))
            })?;

            let pair = (token1_id, token2_id);
            merges.insert(pair, (merge_data.rank, merge_data.new_token_id));
        }

        Ok((vocab, merges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beepe_core::Vocabulary;

    #[test]
    fn test_load_roundtrip() {
        // Create a temporary directory
        let temp_dir = std::env::temp_dir().join("beepe_test_load");
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create a test vocab.json
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello").unwrap();
        vocab.add_token("world").unwrap();

        // Create test merges
        let merges: AHashMap<Pair, (u32, u32)> = AHashMap::new();

        // Save
        use crate::io::save::TokenizerSaver;
        use crate::EncodingMode;
        let _saver = TokenizerSaver::new(&vocab, &merges, EncodingMode::ByteLevel);
        _saver.save(&temp_dir).unwrap();

        // Load
        let (loaded_vocab, loaded_merges) = TokenizerLoader::load(&temp_dir).unwrap();

        assert_eq!(loaded_vocab.len(), vocab.len());
        assert_eq!(loaded_merges.len(), 0);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }
}
