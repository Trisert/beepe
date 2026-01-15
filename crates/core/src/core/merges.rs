//! Merge rule management for BPE.
//!
//! This module provides data structures for storing and accessing BPE merge rules.
//! Merge rules are stored using token IDs rather than strings for fast comparison.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// A pair of token IDs that can be merged.
pub type Pair = (u32, u32);

/// Merge rule mapping: pair -> (rank, new_token_id).
///
/// The rank indicates the priority of this merge rule (lower rank = higher priority).
/// The new_token_id is the ID of the token created by merging this pair.
pub type MergeMap = AHashMap<Pair, (u32, u32)>;

/// Collection of BPE merge rules with efficient lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRules {
    /// Merge rules: pair -> (rank, new_token_id)
    pub merges: MergeMap,
    /// Maximum rank (for validation and ordering)
    pub max_rank: u32,
}

impl MergeRules {
    /// Create a new empty collection of merge rules.
    pub fn new() -> Self {
        Self {
            merges: MergeMap::new(),
            max_rank: 0,
        }
    }

    /// Create a new collection with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            merges: MergeMap::with_capacity(capacity),
            max_rank: 0,
        }
    }

    /// Add a merge rule.
    ///
    /// # Arguments
    /// * `pair` - The pair of token IDs to merge
    /// * `rank` - The priority rank (lower = higher priority)
    /// * `new_token_id` - The ID of the token created by this merge
    pub fn add_merge(&mut self, pair: Pair, rank: u32, new_token_id: u32) {
        self.merges.insert(pair, (rank, new_token_id));
        self.max_rank = self.max_rank.max(rank);
    }

    /// Get the merge rule for a pair.
    ///
    /// Returns Some((rank, new_token_id)) if this pair should be merged,
    /// None otherwise.
    #[inline]
    pub fn get(&self, pair: Pair) -> Option<(u32, u32)> {
        self.merges.get(&pair).copied()
    }

    /// Check if a pair should be merged before another.
    ///
    /// Returns true if `pair` has higher priority (lower rank) than `other`.
    #[inline]
    pub fn should_merge_before(&self, pair: Pair, other: Pair) -> bool {
        match (self.get(pair), self.get(other)) {
            (Some((rank1, _)), Some((rank2, _))) => rank1 < rank2,
            (Some(_), None) => true,
            _ => false,
        }
    }

    /// Get the number of merge rules.
    #[inline]
    pub fn len(&self) -> usize {
        self.merges.len()
    }

    /// Check if there are no merge rules.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }

    /// Create merge rules from a list of pairs.
    ///
    /// The pairs are assigned ranks in order (0, 1, 2, ...).
    pub fn from_pairs(pairs: impl IntoIterator<Item = Pair>) -> Self {
        let mut rules = Self::new();

        for (rank, pair) in pairs.into_iter().enumerate() {
            let rank = rank as u32;
            // The new token ID will be assigned during training
            rules.add_merge(pair, rank, u32::MAX);
        }

        rules
    }
}

impl Default for MergeRules {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about merge rules.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of merge rules
    pub count: usize,
    /// Maximum rank
    pub max_rank: u32,
    /// Minimum rank
    pub min_rank: u32,
}

impl MergeRules {
    /// Get statistics about the merge rules.
    pub fn stats(&self) -> MergeStats {
        let mut min_rank = u32::MAX;
        let mut max_rank = 0;

        for &(rank, _) in self.merges.values() {
            min_rank = min_rank.min(rank);
            max_rank = max_rank.max(rank);
        }

        MergeStats {
            count: self.len(),
            max_rank: self.max_rank,
            min_rank: if min_rank == u32::MAX { 0 } else { min_rank },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_merge() {
        let mut rules = MergeRules::new();
        rules.add_merge((0, 1), 0, 100);
        rules.add_merge((1, 2), 1, 101);

        assert_eq!(rules.get((0, 1)), Some((0, 100)));
        assert_eq!(rules.get((1, 2)), Some((1, 101)));
        assert_eq!(rules.get((2, 3)), None);
    }

    #[test]
    fn test_should_merge_before() {
        let mut rules = MergeRules::new();
        rules.add_merge((0, 1), 0, 100); // Higher priority
        rules.add_merge((1, 2), 1, 101); // Lower priority

        assert!(rules.should_merge_before((0, 1), (1, 2)));
        assert!(!rules.should_merge_before((1, 2), (0, 1)));
        assert!(rules.should_merge_before((0, 1), (2, 3)));
    }

    #[test]
    fn test_from_pairs() {
        let pairs = vec![(0, 1), (1, 2), (2, 3)];
        let rules = MergeRules::from_pairs(pairs);

        assert_eq!(rules.get((0, 1)), Some((0, u32::MAX)));
        assert_eq!(rules.get((1, 2)), Some((1, u32::MAX)));
        assert_eq!(rules.get((2, 3)), Some((2, u32::MAX)));
        assert_eq!(rules.len(), 3);
    }

    #[test]
    fn test_stats() {
        let mut rules = MergeRules::new();
        rules.add_merge((0, 1), 0, 100);
        rules.add_merge((1, 2), 1, 101);
        rules.add_merge((2, 3), 5, 102);

        let stats = rules.stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_rank, 0);
        assert_eq!(stats.max_rank, 5);
    }
}
