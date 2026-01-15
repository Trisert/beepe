//! Training infrastructure for BPE tokenizers.
//!
//! This module provides the training algorithms and utilities for
//! learning BPE merge rules from text data.

pub mod counter;
pub mod trainer;

pub use counter::PairCounter;
pub use trainer::{BpeTrainer, TrainingConfig};
