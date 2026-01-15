//! Beepe-training - BPE training infrastructure
//!
//! This crate provides the training algorithms and utilities for learning
//! BPE merge rules from text data.
//!
//! # Features
//!
//! - Efficient pair frequency counting with parallel processing support
//! - Configurable training parameters (vocab size, min frequency, etc.)
//! - Integration with beepe-core for vocabulary and merge operations
//!
//! # Example
//!
//! ```rust,ignore
//! use beepe_training::{BpeTrainerV2, TrainingConfigV2};
//!
//! let config = TrainingConfigV2 {
//!     vocab_size: 30_000,
//!     min_frequency: 2,
//!     parallel: true,
//!     ..Default::default()
//! };
//!
//! let trainer = BpeTrainerV2::new(config);
//! let (vocab, merges) = trainer.train(&["path/to/text.txt"])?;
//! ```

pub use beepe_core::{Result, TokenizerError};

// Training infrastructure
pub mod training;
pub use training::{BpeTrainerV2, PairCounter, TrainingConfigV2};
