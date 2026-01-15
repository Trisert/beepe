//! Priority queue for BPE merge candidates.
//!
//! This module provides efficient priority queue operations for managing
//! BPE merge candidates during training and encoding.

use crate::core::merges::Pair;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;

/// A merge candidate during BPE training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeCandidate {
    /// The pair of token IDs to merge
    pub pair: Pair,
    /// The frequency/count of this pair
    pub count: u64,
}

impl MergeCandidate {
    /// Create a new merge candidate.
    pub fn new(pair: Pair, count: u64) -> Self {
        Self { pair, count }
    }
}

// Implement Ord for priority queue (higher count = higher priority)
impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher count = higher priority (normal ordering for max-heap)
        self.count
            .cmp(&other.count)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue for BPE merge operations.
///
/// Uses an 8-ary heap for better cache locality than a binary heap.
pub struct PairPriorityQueue {
    /// The heap storing merge candidates
    heap: OctonaryHeap<MergeCandidate>,
    /// Track current counts to detect stale entries
    current_counts: AHashMap<Pair, u64>,
}

impl PairPriorityQueue {
    /// Create a new priority queue with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: OctonaryHeap::with_capacity(capacity),
            current_counts: AHashMap::with_capacity(capacity),
        }
    }

    /// Create a new empty priority queue.
    pub fn new() -> Self {
        Self {
            heap: OctonaryHeap::new(),
            current_counts: AHashMap::new(),
        }
    }

    /// Push a merge candidate onto the queue.
    pub fn push(&mut self, candidate: MergeCandidate) {
        self.current_counts.insert(candidate.pair, candidate.count);
        self.heap.push(candidate);
    }

    /// Pop the highest priority merge candidate.
    ///
    /// Returns None if the queue is empty or only contains stale entries.
    pub fn pop(&mut self) -> Option<MergeCandidate> {
        while let Some(candidate) = self.heap.pop() {
            // Check if entry is stale (count has changed since insertion)
            if let Some(&current) = self.current_counts.get(&candidate.pair) {
                if current == candidate.count {
                    self.current_counts.remove(&candidate.pair);
                    return Some(candidate);
                }
                // Entry is stale, continue to next
            }
        }
        None
    }

    /// Update the count for a pair and push updated candidate.
    ///
    /// This marks any existing entry for the pair as stale.
    pub fn update(&mut self, pair: Pair, new_count: u64) {
        self.current_counts.insert(pair, new_count);
        self.heap.push(MergeCandidate::new(pair, new_count));
    }

    /// Get the number of (potentially stale) entries in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Peek at the highest priority candidate without removing it.
    ///
    /// Returns None if the queue is empty.
    pub fn peek(&self) -> Option<&MergeCandidate> {
        self.heap.peek()
    }

    /// Clear all entries from the queue.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.current_counts.clear();
    }

    /// Get the current count for a pair.
    pub fn get_count(&self, pair: Pair) -> Option<u64> {
        self.current_counts.get(&pair).copied()
    }
}

impl Default for PairPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut queue = PairPriorityQueue::new();

        queue.push(MergeCandidate::new((0, 1), 10));
        queue.push(MergeCandidate::new((1, 2), 20));
        queue.push(MergeCandidate::new((2, 3), 15));

        // Should pop (1, 2) first (highest count: 20)
        let first = queue.pop().unwrap();
        assert_eq!(first.pair, (1, 2));
        assert_eq!(first.count, 20);

        // Then (2, 3) (count: 15)
        let second = queue.pop().unwrap();
        assert_eq!(second.pair, (2, 3));
        assert_eq!(second.count, 15);

        // Then (0, 1) (count: 10)
        let third = queue.pop().unwrap();
        assert_eq!(third.pair, (0, 1));
        assert_eq!(third.count, 10);
    }

    #[test]
    fn test_stale_entry_detection() {
        let mut queue = PairPriorityQueue::new();

        queue.push(MergeCandidate::new((0, 1), 10));
        queue.push(MergeCandidate::new((1, 2), 20));

        // Update count for (0, 1), making first entry stale
        queue.update((0, 1), 15);

        // Pop should return (1, 2) first (count: 20)
        let first = queue.pop().unwrap();
        assert_eq!(first.pair, (1, 2));

        // Then (0, 1) with updated count (15), not stale (10)
        let second = queue.pop().unwrap();
        assert_eq!(second.pair, (0, 1));
        assert_eq!(second.count, 15);

        // Queue should be empty now
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_get_count() {
        let mut queue = PairPriorityQueue::new();

        queue.push(MergeCandidate::new((0, 1), 10));

        assert_eq!(queue.get_count((0, 1)), Some(10));
        assert_eq!(queue.get_count((1, 2)), None);

        // Update count
        queue.update((0, 1), 20);
        assert_eq!(queue.get_count((0, 1)), Some(20));
    }

    #[test]
    fn test_clear() {
        let mut queue = PairPriorityQueue::new();

        queue.push(MergeCandidate::new((0, 1), 10));
        queue.push(MergeCandidate::new((1, 2), 20));

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);

        queue.clear();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }
}
