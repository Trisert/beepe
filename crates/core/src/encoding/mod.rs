//! Encoding modes for BPE tokenization.
//!
//! This module provides different encoding strategies:
//! - Byte-level: tiktoken-style, treating all text as UTF-8 bytes
//! - Character-level: Using Unicode grapheme clusters
//! - Hybrid: Mixed strategies

pub mod byte_level_v2;
pub mod char_level;

pub use byte_level_v2::ByteLevelEncoderV2;
pub use char_level::CharLevelEncoder;
