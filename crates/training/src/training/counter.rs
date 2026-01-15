//! Pair counting for BPE training.
//!
//! This module provides efficient counting of byte/character pair frequencies
//! during BPE training, with support for parallel processing.

use ahash::AHashMap;
use beepe_core::{Pair, Vocabulary};

/// Counter for BPE pair frequencies.
pub struct PairCounter {
    /// Pair -> frequency count
    pair_counts: AHashMap<Pair, u64>,
    /// Word -> tokenized representation (as token IDs)
    words: Vec<Vec<u32>>,
    /// Word -> frequency count
    word_counts: Vec<u64>,
}

impl PairCounter {
    /// Create a new pair counter.
    pub fn new() -> Self {
        Self {
            pair_counts: AHashMap::new(),
            words: Vec::new(),
            word_counts: Vec::new(),
        }
    }

    /// Add text to be processed for pair counting.
    ///
    /// The text will be split into words and each word's token frequencies counted.
    pub fn add_text(&mut self, text: &str, vocab: &Vocabulary) {
        // Split text into words (whitespace separation)
        for word in text.split_whitespace() {
            self.add_word(word, vocab);
        }
    }

    /// Add a single word to the counter.
    pub fn add_word(&mut self, word: &str, vocab: &Vocabulary) {
        // Tokenize word to character sequence
        let word_tokens: Vec<u32> = word
            .chars()
            .map(|c| {
                vocab
                    .get_id(&c.to_string())
                    .unwrap_or_else(|| vocab.get_id("<unk>").unwrap_or(0))
            })
            .collect();

        // Check if we've seen this word before
        if let Some(pos) = self
            .words
            .iter()
            .position(|w| w.as_slice() == word_tokens.as_slice())
        {
            // Increment existing word count
            self.word_counts[pos] += 1;
        } else {
            // Add new word
            self.words.push(word_tokens);
            self.word_counts.push(1);
        }
    }

    /// Count all pairs in parallel.
    ///
    /// This returns a map of pair -> frequency count across all words.
    pub fn count_pairs_parallel(&self) -> AHashMap<Pair, u64> {
        use rayon::prelude::*;

        self.words
            .par_iter()
            .zip(self.word_counts.par_iter())
            .map(|(word, &count)| {
                let mut pair_counts: AHashMap<Pair, u64> = AHashMap::new();

                for window in word.windows(2) {
                    let pair = (window[0], window[1]);
                    *pair_counts.entry(pair).or_insert(0) += count;
                }

                pair_counts
            })
            .reduce(AHashMap::new, |mut acc, pair_counts| {
                for (pair, count) in pair_counts {
                    *acc.entry(pair).or_insert(0) += count;
                }
                acc
            })
    }

    /// Count all pairs sequentially (for debugging or single-threaded use).
    pub fn count_pairs_sequential(&self) -> AHashMap<Pair, u64> {
        let mut pair_counts: AHashMap<Pair, u64> = AHashMap::new();

        for (word, &count) in self.words.iter().zip(self.word_counts.iter()) {
            for window in word.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += count;
            }
        }

        pair_counts
    }

    /// Get the number of unique words.
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get the total count of all word occurrences.
    pub fn total_word_occurrences(&self) -> u64 {
        self.word_counts.iter().sum()
    }

    /// Get a reference to the words.
    pub fn words(&self) -> &[Vec<u32>] {
        &self.words
    }

    /// Get a reference to the word counts.
    pub fn word_counts(&self) -> &[u64] {
        &self.word_counts
    }

    /// Clear all data from the counter.
    pub fn clear(&mut self) {
        self.pair_counts.clear();
        self.words.clear();
        self.word_counts.clear();
    }

    /// Merge a pair in all words (mutates words in place).
    ///
    /// Returns the changes to pair counts as (pair, delta) tuples.
    pub fn merge_pair_in_words(&mut self, pair: Pair, new_token_id: u32) -> Vec<(Pair, i64)> {
        let mut changes: Vec<(Pair, i64)> = Vec::new();

        for word in &mut self.words {
            let mut i = 0;

            while i + 1 < word.len() {
                if word[i] == pair.0 && word[i + 1] == pair.1 {
                    // Record changes to pair counts
                    if i > 0 {
                        let old_pair = (word[i - 1], word[i]);
                        let new_pair = (word[i - 1], new_token_id);
                        changes.push((old_pair, -1));
                        changes.push((new_pair, 1));
                    }
                    if i + 2 < word.len() {
                        let old_pair = (word[i + 1], word[i + 2]);
                        let new_pair = (new_token_id, word[i + 2]);
                        changes.push((old_pair, -1));
                        changes.push((new_pair, 1));
                    }

                    // Perform merge
                    word[i] = new_token_id;
                    word.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        changes
    }
}

impl Default for PairCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beepe_core::Vocabulary;

    #[test]
    fn test_add_word() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("a").unwrap();
        vocab.add_token("b").unwrap();
        vocab.add_token("c").unwrap();

        let mut counter = PairCounter::new();
        counter.add_word("abc", &vocab);

        assert_eq!(counter.word_count(), 1);
        assert_eq!(counter.words()[0].as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_count_pairs_sequential() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("a").unwrap();
        vocab.add_token("b").unwrap();
        vocab.add_token("c").unwrap();

        let mut counter = PairCounter::new();
        counter.add_word("ab", &vocab);
        counter.add_word("bc", &vocab);

        let pairs = counter.count_pairs_sequential();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs.get(&(0, 1)), Some(&1));
        assert_eq!(pairs.get(&(1, 2)), Some(&1));
    }

    #[test]
    fn test_count_pairs_with_frequency() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("a").unwrap();
        vocab.add_token("b").unwrap();

        let mut counter = PairCounter::new();
        counter.add_word("ab", &vocab);
        counter.add_word("ab", &vocab); // Same word twice
        counter.add_word("ab", &vocab); // Three times total

        let pairs = counter.count_pairs_sequential();
        assert_eq!(pairs.get(&(0, 1)), Some(&3));
    }

    #[test]
    fn test_count_pairs_parallel() {
        let mut vocab = Vocabulary::new();
        for c in 'a'..='z' {
            vocab.add_token(&c.to_string()).unwrap();
        }

        let mut counter = PairCounter::new();
        counter.add_word("abc", &vocab);
        counter.add_word("bcd", &vocab);
        counter.add_word("cde", &vocab);

        let pairs = counter.count_pairs_parallel();
        // Should have pairs: (a,b), (b,c), (b,c), (c,d), (c,d), (d,e)
        // Unique: (a,b):1, (b,c):2, (c,d):2, (d,e):1
        assert_eq!(pairs.get(&(0, 1)), Some(&1)); // (a,b)
        assert_eq!(pairs.get(&(1, 2)), Some(&2)); // (b,c)
        assert_eq!(pairs.get(&(2, 3)), Some(&2)); // (c,d)
        assert_eq!(pairs.get(&(3, 4)), Some(&1)); // (d,e)
    }
}
