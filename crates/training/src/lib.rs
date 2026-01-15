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
//! use beepe_training::{BpeTrainer, TrainingConfig};
//!
//! let config = TrainingConfig::builder()
//!     .vocab_size(30_000)
//!     .min_frequency(2)
//!     .build()?;
//!
//! let trainer = BpeTrainer::new(config);
//! let (vocab, merges) = trainer.train(&["path/to/text.txt"])?;
//! ```

pub use beepe_core::{Result, TokenizerError};

// Training infrastructure
pub mod training;
pub use training::{BpeTrainer, BpeTrainerV2, PairCounter, TrainingConfig, TrainingConfigV2};
