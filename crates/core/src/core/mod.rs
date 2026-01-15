//! Core BPE algorithm implementation.
//!
//! This module contains the fundamental data structures and algorithms
//! for byte-pair encoding, independent of any specific encoding mode.

pub mod merges;
pub mod priority;
pub mod vocab;

pub use merges::{MergeMap, MergeRules, Pair};
pub use priority::{MergeCandidate, PairPriorityQueue};
pub use vocab::{SpecialTokens, Vocab, VocabR, Vocabulary};
