//! BPE trainer implementation.
//!
//! # DEPRECATED
//!
//! This module implements a frequency-only BPE training algorithm.
//! Use `trainer_v2::BpeTrainerV2` instead, which provides:
//! - Entropy-weighted merge selection for better compression
//! - Token utility tracking and pruning
//! - Adaptive vocabulary management
//!
//! This module implements the core BPE training algorithm.

use super::counter::PairCounter;
use ahash::AHashMap;
use beepe_core::{MergeCandidate, Pair, PairPriorityQueue, Result, Vocabulary};

/// Configuration for BPE training.
///
/// # DEPRECATED
///
/// Use `TrainingConfigV2` with `BpeTrainerV2` instead for better compression.
#[deprecated(
    since = "0.2.0",
    note = "Use TrainingConfigV2 with BpeTrainerV2 for better compression"
)]
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for a pair to be merged
    pub min_frequency: u64,
    /// Whether to use parallel processing
    pub parallel: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30_000,
            min_frequency: 2,
            parallel: true,
        }
    }
}

/// BPE trainer.
///
/// # DEPRECATED
///
/// Use `BpeTrainerV2` instead for better compression through entropy-weighted
/// merge selection and token utility tracking.
///
/// Trains a BPE tokenizer from text data by iteratively merging the most
/// frequent character/byte pairs.
#[deprecated(since = "0.2.0", note = "Use BpeTrainerV2 for better compression")]
pub struct BpeTrainer {
    /// Configuration
    config: TrainingConfig,
    /// Vocabulary being built
    vocab: Vocabulary,
    /// Merge rules: pair -> (rank, new_token_id)
    merges: AHashMap<Pair, (u32, u32)>,
}

impl BpeTrainer {
    /// Create a new BPE trainer with the given configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            vocab: Vocabulary::new(),
            merges: AHashMap::new(),
        }
    }

    /// Create a new BPE trainer with default configuration.
    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self::new(TrainingConfig {
            vocab_size,
            ..Default::default()
        })
    }

    /// Train the tokenizer on the given text.
    ///
    /// # Arguments
    /// * `text` - Training text data
    ///
    /// # Returns
    /// The trained vocabulary and merge rules
    pub fn train(&mut self, text: &str) -> Result<(Vocabulary, AHashMap<Pair, (u32, u32)>)> {
        // Initialize vocabulary with base characters
        self.initialize_vocab(text)?;

        // Prepare training data
        let mut counter = PairCounter::new();
        counter.add_text(text, &self.vocab);

        // Initial pair counting
        let mut pair_counts = if self.config.parallel {
            counter.count_pairs_parallel()
        } else {
            counter.count_pairs_sequential()
        };

        // Build priority queue
        let mut queue = self.build_queue(&pair_counts);

        // Main training loop: iteratively merge most frequent pairs
        let mut rank = 0u32;

        while self.vocab.len() < self.config.vocab_size && !queue.is_empty() {
            // Get highest priority pair
            let candidate = match queue.pop() {
                Some(c) => c,
                None => break,
            };

            // Check if candidate meets minimum frequency
            if candidate.count < self.config.min_frequency {
                break;
            }

            // Create new token for this merge
            let new_token_id = self.vocab.len() as u32;
            let new_token = self.get_token_string(&candidate.pair);
            self.vocab.add_token(&new_token)?;

            // Add merge rule
            self.merges.insert(candidate.pair, (rank, new_token_id));

            // Update words by merging this pair
            let changes = counter.merge_pair_in_words(candidate.pair, new_token_id);

            // Update pair counts
            self.update_pair_counts(&mut pair_counts, &mut queue, changes, counter.word_counts());

            rank += 1;

            // Stop if we've reached target vocab size
            if self.vocab.len() >= self.config.vocab_size {
                break;
            }
        }

        Ok((self.vocab.clone(), self.merges.clone()))
    }

    /// Initialize vocabulary with all unique characters in the text.
    fn initialize_vocab(&mut self, text: &str) -> Result<()> {
        let mut seen = std::collections::HashSet::new();

        for ch in text.chars() {
            if seen.insert(ch) {
                self.vocab.add_token(&ch.to_string())?;
            }
        }

        Ok(())
    }

    /// Build priority queue from pair counts.
    fn build_queue(&self, pair_counts: &AHashMap<Pair, u64>) -> PairPriorityQueue {
        let mut queue = PairPriorityQueue::with_capacity(pair_counts.len());

        for (&pair, &count) in pair_counts {
            if count >= self.config.min_frequency {
                queue.push(MergeCandidate::new(pair, count));
            }
        }

        queue
    }

    /// Get the token string for a merged pair.
    fn get_token_string(&self, pair: &Pair) -> String {
        let token1 = self.vocab.get_token(pair.0).unwrap_or("");
        let token2 = self.vocab.get_token(pair.1).unwrap_or("");
        format!("{}{}", token1, token2)
    }

    /// Update pair counts after a merge.
    fn update_pair_counts(
        &self,
        pair_counts: &mut AHashMap<Pair, u64>,
        queue: &mut PairPriorityQueue,
        changes: Vec<(Pair, i64)>,
        _word_counts: &[u64],
    ) {
        // Aggregate changes by pair
        let mut aggregated: AHashMap<Pair, i64> = AHashMap::new();
        for (pair, delta) in changes {
            *aggregated.entry(pair).or_insert(0) += delta;
        }

        // Apply changes to pair_counts
        for (pair, delta) in aggregated {
            let current = pair_counts.get(&pair).copied().unwrap_or(0);
            let new_count = (current as i64 + delta).max(0) as u64;

            if new_count > 0 {
                pair_counts.insert(pair, new_count);

                // Update queue if meets minimum frequency
                if new_count >= self.config.min_frequency {
                    queue.push(MergeCandidate::new(pair, new_count));
                }
            } else {
                pair_counts.remove(&pair);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_training() {
        let mut trainer = BpeTrainer::with_vocab_size(100);
        let text = "aaabdaabac";

        let result = trainer.train(text);
        assert!(result.is_ok());

        let (vocab, merges) = result.unwrap();
        assert!(!vocab.is_empty());
        assert!(!merges.is_empty());
    }

    #[test]
    fn test_training_with_simple_text() {
        let mut trainer = BpeTrainer::with_vocab_size(50);
        let text = "hello world hello world";

        let result = trainer.train(text);
        assert!(result.is_ok());

        let (_vocab, merges) = result.unwrap();
        // Should have learned some merges
        assert!(!merges.is_empty());
    }

    #[test]
    fn test_min_frequency_filter() {
        let mut trainer = BpeTrainer::new(TrainingConfig {
            vocab_size: 100,
            min_frequency: 100, // High threshold
            parallel: false,
        });

        let text = "a b c d e";
        let result = trainer.train(text);

        // Should succeed but with fewer merges due to high min_frequency
        assert!(result.is_ok());
    }
}
