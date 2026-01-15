//! Format definitions for tokenizer serialization.
//!
//! This module defines the data structures used for saving/loading
//! tokenizers in various formats.

use serde::{Deserialize, Serialize};

/// Model format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// HuggingFace tokenizer format (vocab.json + merges.txt)
    HuggingFace,
    /// Custom JSON format
    Json,
}

/// HuggingFace format structures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceFormat {
    /// Complete tokenizer data in HF-compatible format
    pub vocab: HuggingFaceVocab,
    pub merges: Vec<MergeRecord>,
}

/// Vocabulary in HuggingFace format.
pub type HuggingFaceVocab = std::collections::HashMap<String, u32>;

/// A single merge record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRecord {
    /// First token in the pair
    pub tokens: [String; 2],
}

/// Merge rule for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMerge {
    /// The pair of tokens being merged
    pub pair: (String, String),
    /// The rank/priority of this merge
    pub rank: u32,
    /// The new token ID created by this merge
    pub new_token_id: u32,
}

/// Complete tokenizer serialization format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTokenizer {
    /// Format version
    pub version: String,
    /// Vocabulary (token -> ID mapping)
    pub vocab: std::collections::HashMap<String, u32>,
    /// Reverse vocabulary (ID -> token mapping)
    pub vocab_r: std::collections::HashMap<u32, String>,
    /// Merge rules
    pub merges: Vec<SerializedMerge>,
    /// Special tokens
    pub special_tokens: SerializedSpecialTokens,
    /// Configuration
    pub config: SerializedConfig,
}

/// Special tokens in serialized format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSpecialTokens {
    pub pad: Option<String>,
    pub unk: Option<String>,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub mask: Option<String>,
    pub user: Option<String>,
    pub assistant: Option<String>,
    pub system: Option<String>,
}

/// Tokenizer configuration in serialized format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedConfig {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub encoding_mode: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_roundtrip() {
        let tokenizer_data = SerializedTokenizer {
            version: "1.0.0".to_string(),
            vocab: {
                let mut map = std::collections::HashMap::new();
                map.insert("hello".to_string(), 0);
                map.insert("world".to_string(), 1);
                map
            },
            vocab_r: {
                let mut map = std::collections::HashMap::new();
                map.insert(0, "hello".to_string());
                map.insert(1, "world".to_string());
                map
            },
            merges: vec![SerializedMerge {
                pair: ("h".to_string(), "e".to_string()),
                rank: 0,
                new_token_id: 2,
            }],
            special_tokens: SerializedSpecialTokens {
                pad: Some("<pad>".to_string()),
                unk: Some("<unk>".to_string()),
                bos: Some("<bos>".to_string()),
                eos: Some("<eos>".to_string()),
                mask: None,
                user: None,
                assistant: None,
                system: None,
            },
            config: SerializedConfig {
                vocab_size: 10000,
                min_frequency: 2,
                encoding_mode: "ByteLevel".to_string(),
            },
        };

        // Serialize
        let json = serde_json::to_string(&tokenizer_data).unwrap();
        println!("{}", json);

        // Deserialize
        let deserialized: SerializedTokenizer = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.version, tokenizer_data.version);
        assert_eq!(deserialized.vocab, tokenizer_data.vocab);
        assert_eq!(deserialized.merges.len(), tokenizer_data.merges.len());
    }
}
