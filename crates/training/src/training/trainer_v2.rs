//! Entropy-weighted BPE trainer implementation.
//!
//! This module implements an improved BPE training algorithm that considers
//! not just frequency, but also information content (entropy) and token utility.
//! This results in better compression by prioritizing merges that reduce
//! uncertainty in the text.

use super::counter::PairCounter;
use ahash::AHashMap;
use beepe_core::{MergeCandidate, Pair, PairPriorityQueue, Result, Vocabulary};
use std::collections::HashSet;

/// Configuration for entropy-weighted BPE training.
#[derive(Debug, Clone)]
pub struct TrainingConfigV2 {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for a pair to be merged
    pub min_frequency: u64,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Weight for frequency component in merge score (0.0-1.0)
    pub frequency_weight: f32,
    /// Weight for entropy reduction component (0.0-1.0)
    pub entropy_weight: f32,
    /// Weight for context diversity component (0.0-1.0)
    pub diversity_weight: f32,
    /// Threshold for token utility pruning (0.0-1.0)
    pub utility_threshold: f32,
    /// Maximum number of low-utility tokens to prune per iteration
    pub prune_batch_size: usize,
}

impl Default for TrainingConfigV2 {
    fn default() -> Self {
        Self {
            vocab_size: 30_000,
            min_frequency: 2,
            parallel: true,
            frequency_weight: 0.6,
            entropy_weight: 0.3,
            diversity_weight: 0.1,
            utility_threshold: 0.01,
            prune_batch_size: 100,
        }
    }
}

/// Entropy-weighted BPE trainer.
///
/// Trains a BPE tokenizer by considering frequency, information content,
/// and token utility, resulting in better compression ratios.
pub struct BpeTrainerV2 {
    /// Configuration
    config: TrainingConfigV2,
    /// Vocabulary being built
    vocab: Vocabulary,
    /// Merge rules: pair -> (rank, new_token_id)
    merges: AHashMap<Pair, (u32, u32)>,
    /// Character entropy cache
    char_entropy: AHashMap<char, f32>,
    /// Token utility scores
    token_utility: AHashMap<u32, f32>,
    /// Context diversity tracking
    pair_contexts: AHashMap<Pair, HashSet<Pair>>,
}

impl BpeTrainerV2 {
    /// Create a new entropy-weighted BPE trainer.
    pub fn new(config: TrainingConfigV2) -> Self {
        Self {
            config,
            vocab: Vocabulary::new(),
            merges: AHashMap::new(),
            char_entropy: AHashMap::new(),
            token_utility: AHashMap::new(),
            pair_contexts: AHashMap::new(),
        }
    }

    /// Create a new trainer with default configuration.
    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self::new(TrainingConfigV2 {
            vocab_size,
            ..Default::default()
        })
    }

    /// Train the tokenizer on the given text using entropy-weighted selection.
    ///
    /// # Arguments
    /// * `text` - Training text data
    ///
    /// # Returns
    /// The trained vocabulary and merge rules
    pub fn train(&mut self, text: &str) -> Result<(Vocabulary, AHashMap<Pair, (u32, u32)>)> {
        // Initialize vocabulary with base characters
        self.initialize_vocab(text)?;

        // Calculate character entropy for the training text
        self.calculate_char_entropy(text)?;

        // Prepare training data
        let mut counter = PairCounter::new();
        counter.add_text(text, &self.vocab);

        // Initial pair counting
        let mut pair_counts = if self.config.parallel {
            counter.count_pairs_parallel()
        } else {
            counter.count_pairs_sequential()
        };

        // Build priority queue with entropy-weighted scoring
        let mut queue = self.build_entropy_weighted_queue(&pair_counts);

        // Track context diversity
        self.track_context_diversity(&counter);

        // Main training loop
        let mut rank = 0u32;
        let mut prune_counter = 0u32;

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

            // Skip if this pair was already processed (stale entry)
            if self.merges.contains_key(&candidate.pair) {
                continue;
            }

            // Create new token for this merge
            let new_token_id = self.vocab.len() as u32;
            let new_token = self.get_token_string(candidate.pair);
            self.vocab.add_token(&new_token)?;

            // Calculate and track token utility
            let utility = self.calculate_token_utility(candidate.pair, candidate.count);
            self.token_utility.insert(new_token_id, utility);

            // Add merge rule
            self.merges.insert(candidate.pair, (rank, new_token_id));

            // Update words by merging this pair
            let changes = counter.merge_pair_in_words(candidate.pair, new_token_id);

            // Update pair counts
            self.update_pair_counts(&mut pair_counts, &mut queue, changes, counter.word_counts());

            rank += 1;

            // Periodically prune low-utility tokens
            prune_counter += 1;
            if prune_counter >= self.config.prune_batch_size as u32 {
                self.prune_low_utility_tokens(&mut counter, &mut pair_counts, &mut queue);
                prune_counter = 0;
            }

            // Stop if we've reached target vocab size
            if self.vocab.len() >= self.config.vocab_size {
                break;
            }
        }

        // Final pruning of very low utility tokens
        self.final_pruning(&mut counter);

        Ok((self.vocab.clone(), self.merges.clone()))
    }

    /// Calculate character entropy for the training text.
    ///
    /// High entropy characters are more informative and should be prioritized.
    fn calculate_char_entropy(&mut self, text: &str) -> Result<()> {
        // Count character frequencies
        let mut char_counts: AHashMap<char, u64> = AHashMap::new();
        let mut total_chars = 0u64;

        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
            total_chars += 1;
        }

        // Calculate entropy for each character
        for (&ch, &count) in &char_counts {
            let probability = count as f64 / total_chars as f64;
            let entropy = -probability * probability.log2();
            self.char_entropy.insert(ch, entropy as f32);
        }

        Ok(())
    }

    /// Build priority queue with entropy-weighted scoring.
    fn build_entropy_weighted_queue(&self, pair_counts: &AHashMap<Pair, u64>) -> PairPriorityQueue {
        let mut queue = PairPriorityQueue::with_capacity(pair_counts.len());

        for (&pair, &count) in pair_counts {
            if count >= self.config.min_frequency {
                // Calculate entropy-weighted score
                let score = self.calculate_merge_score(pair, count);
                // Use the score as the count for ordering (higher is better)
                // Note: This means we're ordering by score, not raw frequency
                let adjusted_count = (score * 1000.0) as u64;
                queue.push(MergeCandidate::new(pair, adjusted_count));
            }
        }

        queue
    }

    /// Calculate merge score combining frequency, entropy, and diversity.
    ///
    /// Formula: frequency_weight * log(freq + 1)
    ///         + entropy_weight * entropy_reduction
    ///         + diversity_weight * context_diversity
    fn calculate_merge_score(&self, pair: Pair, frequency: u64) -> f32 {
        // Frequency component (log-scale)
        let freq_score = (frequency as f32 + 1.0).ln_1p();

        // Entropy reduction component
        let entropy_score = self.entropy_reduction_score(pair);

        // Context diversity component
        let diversity_score = self.context_diversity_score(pair);

        // Weighted combination
        self.config.frequency_weight * freq_score
            + self.config.entropy_weight * entropy_score
            + self.config.diversity_weight * diversity_score
    }

    /// Calculate entropy reduction score for a pair.
    ///
    /// High score means merging this pair reduces uncertainty significantly.
    fn entropy_reduction_score(&self, pair: Pair) -> f32 {
        let token1_entropy = self.get_token_entropy(pair.0);
        let token2_entropy = self.get_token_entropy(pair.1);

        // The score is higher when both tokens have high entropy
        // (merging high-entropy tokens reduces more uncertainty)
        (token1_entropy + token2_entropy) / 2.0
    }

    /// Get the entropy of a token by averaging character entropy.
    fn get_token_entropy(&self, token_id: u32) -> f32 {
        if let Some(token_str) = self.vocab.get_token(token_id) {
            let chars: Vec<char> = token_str.chars().collect();
            if chars.is_empty() {
                return 0.0;
            }

            let total_entropy: f32 = chars
                .iter()
                .map(|&ch| self.char_entropy.get(&ch).copied().unwrap_or(0.0))
                .sum();

            total_entropy / chars.len() as f32
        } else {
            0.0
        }
    }

    /// Calculate context diversity score for a pair.
    ///
    /// Higher score means the pair appears in many different contexts.
    fn context_diversity_score(&self, pair: Pair) -> f32 {
        self.pair_contexts
            .get(&pair)
            .map(|contexts| contexts.len() as f32)
            .unwrap_or(0.0)
            .ln_1p()
    }

    /// Track context diversity for all pairs.
    fn track_context_diversity(&mut self, counter: &PairCounter) {
        // For each pair, track what other pairs appear nearby
        // This is a simplified version - a full implementation would
        // track left and right contexts separately
        for word in counter.words() {
            // Extract all pairs from the word using windows
            let pairs: Vec<Pair> = word.windows(2).map(|w| (w[0], w[1])).collect();
            for i in 0..pairs.len() {
                for j in (i + 1)..pairs.len().min(i + 3) {
                    let entry = self
                        .pair_contexts
                        .entry(pairs[i])
                        .or_insert_with(HashSet::new);
                    entry.insert(pairs[j]);
                    let entry = self
                        .pair_contexts
                        .entry(pairs[j])
                        .or_insert_with(HashSet::new);
                    entry.insert(pairs[i]);
                }
            }
        }
    }

    /// Calculate utility score for a token.
    ///
    /// Utility considers:
    /// - How much it reduces sequence length
    /// - How frequently it's used
    fn calculate_token_utility(&self, pair: Pair, frequency: u64) -> f32 {
        // Base utility from frequency
        let freq_utility = (frequency as f32).ln_1p();

        // Bonus for longer tokens (more compression potential)
        let token1 = self.vocab.get_token(pair.0).map_or("", |s| s);
        let token2 = self.vocab.get_token(pair.1).map_or("", |s| s);
        let length_bonus = (token1.len() + token2.len()) as f32;

        freq_utility * (1.0 + length_bonus / 10.0)
    }

    /// Prune low-utility tokens to free up vocabulary budget.
    fn prune_low_utility_tokens(
        &mut self,
        counter: &mut PairCounter,
        pair_counts: &mut AHashMap<Pair, u64>,
        queue: &mut PairPriorityQueue,
    ) {
        // Find tokens with utility below threshold
        let low_utility_tokens: Vec<u32> = self
            .token_utility
            .iter()
            .filter(|(_, &utility)| utility < self.config.utility_threshold)
            .map(|(&id, _)| id)
            .take(self.config.prune_batch_size)
            .collect();

        // Remove low-utility tokens from vocab and merges
        for token_id in low_utility_tokens {
            // Find and remove merge rules that create this token
            self.merges.retain(|_, &mut (_, new_id)| new_id != token_id);
            self.token_utility.remove(&token_id);

            // Note: We don't remove from vocab to avoid ID reassignment issues
            // In a full implementation, we'd need to rebuild the vocab
        }

        // Recalculate pair counts after pruning
        // This is expensive, so we only do it periodically
        *pair_counts = if self.config.parallel {
            counter.count_pairs_parallel()
        } else {
            counter.count_pairs_sequential()
        };

        // Rebuild queue with updated scores
        *queue = self.build_entropy_weighted_queue(pair_counts);
    }

    /// Final pruning pass to remove very low utility tokens.
    fn final_pruning(&mut self, _counter: &mut PairCounter) {
        // Find the bottom 1% of tokens by utility
        let mut utilities: Vec<_> = self.token_utility.iter().collect();
        utilities.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        let prune_count = (utilities.len() / 100).max(1);
        // Collect token IDs to remove first
        let to_remove: Vec<u32> = utilities
            .iter()
            .take(prune_count)
            .map(|(&id, _)| id)
            .collect();
        for token_id in to_remove {
            self.token_utility.remove(&token_id);
            // Keep merge rules for consistency, but mark tokens as low utility
        }
    }

    /// Initialize vocabulary with all unique characters in the text.
    fn initialize_vocab(&mut self, text: &str) -> Result<()> {
        let mut seen = HashSet::new();

        for ch in text.chars() {
            if seen.insert(ch) {
                self.vocab.add_token(&ch.to_string())?;
            }
        }

        Ok(())
    }

    /// Get the token string for a merged pair.
    fn get_token_string(&self, pair: Pair) -> String {
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

                // Update queue with new entropy-weighted score
                if new_count >= self.config.min_frequency {
                    let score = self.calculate_merge_score(pair, new_count);
                    // Use the score as the count for ordering
                    let adjusted_count = (score * 1000.0) as u64;
                    queue.push(MergeCandidate::new(pair, adjusted_count));
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
    fn test_entropy_calculation() {
        let mut trainer = BpeTrainerV2::with_vocab_size(100);
        let text = "hello world";

        trainer.calculate_char_entropy(text).unwrap();

        // Check that we have entropy values
        assert!(!trainer.char_entropy.is_empty());

        // Space should have relatively low entropy (common)
        let space_entropy = trainer.char_entropy.get(&' ').copied().unwrap_or(0.0);
        assert!(space_entropy > 0.0);
    }

    #[test]
    fn test_merge_score_calculation() {
        let mut trainer = BpeTrainerV2::with_vocab_size(100);

        // Add some characters to vocab
        trainer.vocab.add_token("h").unwrap();
        trainer.vocab.add_token("e").unwrap();

        let pair = (0, 1); // "h" + "e"
        let frequency = 100;

        let score = trainer.calculate_merge_score(pair, frequency);

        // Score should be positive
        assert!(score > 0.0);
    }

    #[test]
    fn test_basic_training() {
        let mut trainer = BpeTrainerV2::with_vocab_size(50);
        let text = "hello world hello world";

        let (vocab, merges) = trainer.train(text).unwrap();

        // Should have learned something
        assert!(vocab.len() > 10);
        assert!(merges.len() > 0);
    }
}
