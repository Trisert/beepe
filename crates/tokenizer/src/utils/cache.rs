//! Encoding cache for repeated text sequences.
//!
//! This module provides an LRU cache for storing recently encoded sequences,
//! significantly improving performance for repeated text.

use crate::Result;
use std::collections::HashMap;

/// LRU cache for encoding results.
///
/// Uses a simple HashMap-based implementation with a fixed capacity.
/// When the cache exceeds capacity, the oldest entries are evicted.
pub struct EncodingCache {
    /// The cache storing text -> encoded tokens
    cache: HashMap<String, Vec<u32>>,
    /// Maximum number of entries in the cache
    capacity: usize,
    /// Track insertion order for LRU eviction
    insertion_order: Vec<String>,
}

impl EncodingCache {
    /// Create a new encoding cache with the given capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of entries to store
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
            insertion_order: Vec::with_capacity(capacity),
        }
    }

    /// Create a new encoding cache with default capacity (1000).
    pub fn new() -> Self {
        Self::with_capacity(1000)
    }

    /// Get cached encoding or compute using the provided function.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `encoder` - Function to compute encoding if not cached
    ///
    /// # Returns
    /// The encoded token IDs
    pub fn get_or_encode<F>(&mut self, text: &str, encoder: F) -> Result<Vec<u32>>
    where
        F: FnOnce(&str) -> Result<Vec<u32>>,
    {
        // Check if we have a cached result
        if let Some(cached) = self.cache.get(text).cloned() {
            // Update insertion order (move to end = most recently used)
            if let Some(pos) = self.insertion_order.iter().position(|x| x == text) {
                self.insertion_order.remove(pos);
            }
            self.insertion_order.push(text.to_string());
            return Ok(cached);
        }

        // Not cached, encode the text
        let encoded = encoder(text)?;

        // Add to cache
        self.insert(text.to_string(), encoded.clone());

        Ok(encoded)
    }

    /// Insert a value into the cache.
    fn insert(&mut self, key: String, value: Vec<u32>) {
        // If at capacity, evict oldest entry
        if self.insertion_order.len() >= self.capacity && !self.cache.contains_key(&key) {
            if let Some(oldest) = self.insertion_order.first() {
                self.cache.remove(oldest);
                self.insertion_order.remove(0);
            }
        }

        // If key already exists, remove old entry
        if self.cache.contains_key(&key) {
            if let Some(pos) = self.insertion_order.iter().position(|x| x == &key) {
                self.insertion_order.remove(pos);
            }
        }

        // Insert new entry
        self.cache.insert(key.clone(), value);
        self.insertion_order.push(key);
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.insertion_order.clear();
    }

    /// Get the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the cache capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resize the cache.
    ///
    /// If the new capacity is smaller than the current size,
    /// oldest entries will be evicted.
    pub fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;

        // Evict oldest entries if necessary
        while self.insertion_order.len() > new_capacity {
            if let Some(oldest) = self.insertion_order.first() {
                self.cache.remove(oldest);
                self.insertion_order.remove(0);
            }
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            capacity: self.capacity,
            hit_rate: None, // Could track hits/misses if needed
        }
    }
}

impl Default for EncodingCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of entries
    pub entries: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Cache hit rate (None if not tracked)
    pub hit_rate: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_miss() {
        let mut cache = EncodingCache::with_capacity(3);

        // First call should miss (encode)
        let result1 = cache
            .get_or_encode("hello", |_text| Ok(vec![1, 2, 3]))
            .unwrap();
        assert_eq!(result1, vec![1, 2, 3]);

        // Second call should hit (cached)
        let result2 = cache
            .get_or_encode("hello", |_| panic!("Should not encode"))
            .unwrap();
        assert_eq!(result2, vec![1, 2, 3]);

        // Different text should miss
        let result3 = cache
            .get_or_encode("world", |_text| Ok(vec![4, 5]))
            .unwrap();
        assert_eq!(result3, vec![4, 5]);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EncodingCache::with_capacity(2);

        cache.get_or_encode("a", |_| Ok(vec![1])).unwrap();
        cache.get_or_encode("b", |_| Ok(vec![2])).unwrap();
        cache.get_or_encode("c", |_| Ok(vec![3])).unwrap();

        // "a" should have been evicted (oldest)
        assert!(cache.cache.get("a").is_none());
        assert!(cache.cache.get("b").is_some());
        assert!(cache.cache.get("c").is_some());
    }

    #[test]
    fn test_lru_update() {
        let mut cache = EncodingCache::with_capacity(2);

        cache.get_or_encode("a", |_| Ok(vec![1])).unwrap();
        cache.get_or_encode("b", |_| Ok(vec![2])).unwrap();

        // Access "a" to make it recently used
        cache.get_or_encode("a", |_| Ok(vec![1])).unwrap();

        // Add "c" - should evict "b" not "a"
        cache.get_or_encode("c", |_| Ok(vec![3])).unwrap();

        assert!(cache.cache.get("a").is_some());
        assert!(cache.cache.get("b").is_none());
        assert!(cache.cache.get("c").is_some());
    }

    #[test]
    fn test_clear() {
        let mut cache = EncodingCache::new();

        cache.get_or_encode("hello", |_| Ok(vec![1, 2, 3])).unwrap();
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_resize() {
        let mut cache = EncodingCache::with_capacity(5);

        cache.get_or_encode("a", |_| Ok(vec![1])).unwrap();
        cache.get_or_encode("b", |_| Ok(vec![2])).unwrap();
        cache.get_or_encode("c", |_| Ok(vec![3])).unwrap();

        assert_eq!(cache.len(), 3);

        // Resize to 2 - should evict oldest entries
        cache.resize(2);
        assert_eq!(cache.len(), 2);
        assert!(cache.cache.get("a").is_none()); // Oldest evicted
        assert!(cache.cache.get("b").is_some()); // Second oldest remains
        assert!(cache.cache.get("c").is_some()); // Newest remains
    }

    #[test]
    fn test_stats() {
        let cache = EncodingCache::with_capacity(100);
        let stats = cache.stats();

        assert_eq!(stats.entries, 0);
        assert_eq!(stats.capacity, 100);
    }
}
