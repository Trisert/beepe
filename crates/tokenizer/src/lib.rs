//! Beepe-tokenizer - High-level tokenizer API
//!
//! This crate provides a user-friendly interface for BPE tokenization,
//! integrating all components (vocabulary, merge rules, encoder) into
//! a single, easy-to-use API.
//!
//! # Features
//!
//! - Simple builder pattern for tokenizer configuration
//! - Support for multiple encoding modes (byte-level, character-level)
//! - Pre-tokenization pipeline (splitting, normalization, byte fallback)
//! - LLM-specific features (special tokens, chat templates, role tokens)
//! - Loading and saving in various formats (HuggingFace, tiktoken)
//!
//! # Example
//!
//! ```rust
//! use beepe_tokenizer::Tokenizer;
//!
//! // Build a tokenizer with configuration
//! let tokenizer = Tokenizer::builder()
//!     .vocab_size(30_000)
//!     .build()?;
//!
//! // Encode text
//! let encoding = tokenizer.encode("Hello, world!", true)?;
//! println!("{:?}", encoding.ids);
//!
//! // Decode tokens
//! let text = tokenizer.decode(&encoding.ids, false);
//! # Ok::<(), beepe_tokenizer::TokenizerError>(())
//! ```

// Re-export core types
pub use beepe_core::{Result, SpecialTokensConfig, TokenizerError};

// Tokenizer API
pub mod tokenizer;
pub use tokenizer::{Encoding, EncodingMode, Tokenizer, TokenizerBuilder};

// IO/Serialization
pub mod io;
pub use io::{HuggingFaceFormat, ModelFormat, TokenizerLoader, TokenizerSaver};

// Pre-tokenization
pub mod pre_tokenizer;
pub use pre_tokenizer::{ByteFallback, Normalizer, Splitter};

// Utilities
pub mod utils;
pub use utils::EncodingCache;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
