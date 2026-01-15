//! Error types for the BPE tokenizer library.

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for the tokenizer library.
#[derive(Error, Debug)]
pub enum TokenizerError {
    /// Error during tokenization
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Error during training
    #[error("Training error: {0}")]
    Training(String),

    /// Error loading vocabulary or merges
    #[error("Load error: {0}")]
    Load(String),

    /// Error saving vocabulary or merges
    #[error("Save error: {0}")]
    Save(String),

    /// I/O error with file context
    #[error("I/O error for {path}: {err}")]
    Io {
        path: PathBuf,
        #[source]
        err: std::io::Error,
    },

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Unknown token ID
    #[error("Unknown token ID: {0}")]
    UnknownTokenId(u32),

    /// Unknown token string
    #[error("Unknown token: {0}")]
    UnknownToken(String),

    /// Unknown role token
    #[error("Unknown role token: {0}")]
    UnknownRole(String),

    /// Vocabulary overflow
    #[error("Vocabulary size exceeded maximum of {max} (tried to add {tried})")]
    VocabularyOverflow { max: usize, tried: usize },

    /// Invalid merge rule
    #[error("Invalid merge rule: {0}")]
    InvalidMerge(String),
}

/// Result type alias for tokenizer operations.
pub type Result<T> = std::result::Result<T, TokenizerError>;
