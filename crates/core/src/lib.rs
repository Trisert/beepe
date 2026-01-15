//! Beepe-core - Core BPE algorithm implementation
//!
//! This crate provides the fundamental data structures and algorithms for
//! byte-pair encoding (BPE), independent of any specific encoding mode.
//!
//! # Features
//!
//! - Efficient vocabulary storage using `AHashMap` and compact strings
//! - Fast merge rule lookups and priority queue operations
//! - Support for special tokens and role tokens
//! - Error handling with detailed diagnostics
//!
//! # Example
//!
//! ```rust
//! use beepe_core::{Vocabulary, Vocab};
//!
//! // Create a new vocabulary
//! let mut vocab = Vocabulary::new();
//! vocab.add_token("hello");
//! vocab.add_token("world");
//! ```

pub mod error;
pub use error::{Result, TokenizerError};

// Core BPE algorithm modules
pub mod core;
pub use core::vocab::SpecialTokensConfig;
pub use core::{
    MergeCandidate, MergeMap, MergeRules, Pair, PairPriorityQueue, SpecialTokens, Vocab, VocabR,
    Vocabulary,
};

// Unified vocabulary (memory-efficient v2)
pub mod vocab_v2;
pub use vocab_v2::UnifiedVocab;

// Encoding modes
pub mod encoding;
pub use encoding::{ByteLevelEncoderV2, CharLevelEncoder};
