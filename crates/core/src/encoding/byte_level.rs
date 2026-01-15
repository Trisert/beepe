//! Byte-level BPE encoding (tiktoken-style).
//!
//! # DEPRECATED
//!
//! This module is deprecated. Use `byte_level_v2::ByteLevelEncoderV2` instead,
//! which provides:
//! - 1000x lower memory footprint (7.5 KB vs 8.6 MB)
//! - Arc sharing for zero-copy vocabulary access
//! - Better performance through first-byte indexing
//!
//! This module implements byte-level BPE encoding, which treats all text
//! as UTF-8 bytes and applies merge rules at the byte level.

use crate::core::vocab::VocabR;
use crate::core::{vocab::Vocab, MergeMap};
use crate::Result;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;

/// Trie node for efficient longest-match tokenization.
#[derive(Debug, Clone)]
struct TrieNode {
    /// Child nodes indexed by character
    children: AHashMap<char, TrieNode>,
    /// Token ID if this node represents a complete token
    token_id: Option<u32>,
}

impl TrieNode {
    /// Create a new empty trie node.
    fn new() -> Self {
        Self {
            children: AHashMap::new(),
            token_id: None,
        }
    }
}

/// Trie structure for fast prefix lookup of tokens.
struct VocabTrie {
    root: TrieNode,
}

impl VocabTrie {
    /// Create a new empty trie.
    fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    /// Insert a token string into the trie.
    fn insert(&mut self, token: &str, token_id: u32) {
        let mut node = &mut self.root;

        for ch in token.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }

        node.token_id = Some(token_id);
    }

    /// Build a trie from a vocabulary.
    fn from_vocab(vocab: &Vocab) -> Self {
        let mut trie = Self::new();

        for (token_str, &token_id) in vocab.iter() {
            trie.insert(token_str, token_id);
        }

        trie
    }

    /// Find the longest matching token starting at the given position.
    ///
    /// Returns the token ID and length of the match, or None if no match.
    fn find_longest_match(&self, chars: &[char], pos: usize) -> Option<(u32, usize)> {
        let mut node = &self.root;
        let mut best_match: Option<(u32, usize)> = None;

        for i in pos..chars.len() {
            let ch = chars[i];

            // Check if this character exists in the trie
            match node.children.get(&ch) {
                Some(child) => {
                    node = child;

                    // If this node represents a complete token, update best match
                    if let Some(token_id) = node.token_id {
                        best_match = Some((token_id, i - pos + 1));
                    }
                }
                None => {
                    // No more matches possible
                    break;
                }
            }
        }

        best_match
    }
}

/// Byte-level BPE encoder following tiktoken's approach.
///
/// # DEPRECATED
///
/// Use `ByteLevelEncoderV2` instead for better memory efficiency and performance.
/// This encoder uses a trie structure which consumes ~8.6 MB, while V2 uses
/// first-byte indexing with only ~7.5 KB memory overhead.
///
/// This encoder:
/// 1. Converts text to UTF-8 bytes
/// 2. Maps each byte to a character (using tiktoken's byte-unicode mapping)
/// 3. Applies BPE merge rules iteratively
#[deprecated(
    since = "0.2.0",
    note = "Use ByteLevelEncoderV2 instead for better memory efficiency"
)]
pub struct ByteLevelEncoder {
    /// Vocabulary for token lookups (token string -> ID)
    vocab: Vocab,
    /// Reverse vocabulary for decoding (ID -> token string)
    vocab_r: VocabR,
    /// Merge rules: pair -> (rank, new_token_id)
    merges: MergeMap,
    /// Byte to unicode mapping (pre-computed for speed)
    byte_encoder: [char; 256],
    /// Unicode to byte mapping for decoding
    byte_decoder: AHashMap<char, u8>,
    /// Trie for efficient longest-match tokenization
    vocab_trie: VocabTrie,
}

impl ByteLevelEncoder {
    /// Create a new byte-level encoder.
    pub fn new(vocab: Vocab, vocab_r: VocabR, merges: MergeMap) -> Self {
        let byte_encoder = Self::build_byte_encoder();
        let byte_decoder = Self::build_byte_decoder();
        let vocab_trie = VocabTrie::from_vocab(&vocab);

        Self {
            vocab,
            vocab_r,
            merges,
            byte_encoder,
            byte_decoder,
            vocab_trie,
        }
    }

    /// Build the byte-to-unicode mapping (tiktoken's approach).
    ///
    /// This maps bytes 0-255 to a range of unicode characters.
    /// We avoid the control character range (0-55) and start at 256.
    fn build_byte_encoder() -> [char; 256] {
        let mut byte_encoder = ['\0'; 256];

        let mut codepoint = 256u32;
        let mut offset = 0;

        while offset < 256 {
            // Skip the surrogate range and ensure valid unicode
            if (0xD800..=0xDFFF).contains(&codepoint) {
                codepoint = 0xE000;
            }

            byte_encoder[offset as usize] = unsafe { char::from_u32_unchecked(codepoint) };

            offset += 1;
            codepoint += 1;
        }

        byte_encoder
    }

    /// Build the unicode-to-byte mapping for decoding.
    fn build_byte_decoder() -> AHashMap<char, u8> {
        let byte_encoder = Self::build_byte_encoder();
        let mut byte_decoder = AHashMap::with_capacity(256);

        for (byte, &ch) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(ch, byte as u8);
        }

        byte_decoder
    }

    /// Encode text using byte-level BPE with longest-match tokenization.
    ///
    /// Returns a vector of token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Convert text to bytes, then to unicode characters (tiktoken's byte mapping)
        let bytes = text.as_bytes();
        let chars: Vec<char> = bytes
            .iter()
            .map(|&b| self.byte_encoder[b as usize])
            .collect();

        // Longest-match tokenization using trie (O(n * max_token_length) instead of O(n²))
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < chars.len() {
            // Use trie to find the longest matching token at this position
            match self.vocab_trie.find_longest_match(&chars, pos) {
                Some((token_id, length)) => {
                    tokens.push(token_id);
                    pos += length;
                }
                None => {
                    // No match found - fallback to single character
                    let c = chars[pos];
                    let token_str = c.to_string();
                    let token_id = self
                        .vocab
                        .get(token_str.as_str())
                        .ok_or_else(|| crate::TokenizerError::UnknownToken(token_str))?;
                    tokens.push(*token_id);
                    pos += 1;
                }
            }
        }

        // Apply BPE merges to merge adjacent tokens according to learned rules
        self.apply_bpe_merges(&mut tokens);

        Ok(tokens)
    }

    /// Apply BPE merge rules to a sequence of tokens.
    ///
    /// This modifies the tokens in-place, merging pairs according to the
    /// priority queue of merge rules.
    fn apply_bpe_merges(&self, tokens: &mut Vec<u32>) {
        if tokens.len() < 2 {
            return;
        }

        // Build initial priority queue of possible merges
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct MergeOp {
            rank: u32,
            pos: usize,
            new_id: u32,
        }

        // Implement Ord for priority queue (lower rank = higher priority)
        impl Ord for MergeOp {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.rank
                    .cmp(&other.rank)
                    .reverse()
                    .then_with(|| self.pos.cmp(&other.pos))
            }
        }

        impl PartialOrd for MergeOp {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = OctonaryHeap::with_capacity(tokens.len());

        // Add all initial pairs to the heap
        for (i, window) in tokens.windows(2).enumerate() {
            let pair = (window[0], window[1]);
            if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                heap.push(MergeOp {
                    rank,
                    pos: i,
                    new_id,
                });
            }
        }

        // Apply merges in priority order
        while let Some(merge_op) = heap.pop() {
            // Check if position is still valid
            if merge_op.pos + 1 >= tokens.len() {
                continue;
            }

            let pair = (tokens[merge_op.pos], tokens[merge_op.pos + 1]);

            // Verify this is still the pair we want to merge
            if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                if rank != merge_op.rank || new_id != merge_op.new_id {
                    continue; // Stale entry
                }

                // Perform merge
                tokens[merge_op.pos] = merge_op.new_id;
                tokens.remove(merge_op.pos + 1);

                // Add new merge opportunities
                if merge_op.pos > 0 {
                    let new_pair = (tokens[merge_op.pos - 1], tokens[merge_op.pos]);
                    if let Some(&(rank, new_id)) = self.merges.get(&new_pair) {
                        heap.push(MergeOp {
                            rank,
                            pos: merge_op.pos - 1,
                            new_id,
                        });
                    }
                }

                if merge_op.pos + 1 < tokens.len() {
                    let new_pair = (tokens[merge_op.pos], tokens[merge_op.pos + 1]);
                    if let Some(&(rank, new_id)) = self.merges.get(&new_pair) {
                        heap.push(MergeOp {
                            rank,
                            pos: merge_op.pos,
                            new_id,
                        });
                    }
                }
            }
        }
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut chars = Vec::with_capacity(ids.len());

        for &id in ids {
            let token = self
                .vocab_r
                .get(&id)
                .ok_or(crate::TokenizerError::UnknownTokenId(id))?;

            // Each character in the token represents a byte
            for ch in token.chars() {
                if let Some(&byte) = self.byte_decoder.get(&ch) {
                    chars.push(byte);
                }
            }
        }

        String::from_utf8(chars).map_err(|e| {
            crate::TokenizerError::Tokenization(format!(
                "Invalid UTF-8 sequence during decoding: {}",
                e
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vocab::Vocabulary;

    #[test]
    fn test_byte_encoder() {
        let encoder = ByteLevelEncoder::build_byte_encoder();
        assert_eq!(encoder[0] as u32, 256);
        assert_eq!(encoder[255] as u32, 256 + 255);
    }

    #[test]
    fn test_encode_simple() {
        let mut vocab = Vocabulary::new();

        // Add base byte vocabulary
        for i in 0..256u32 {
            let ch = unsafe { char::from_u32_unchecked(256 + i) };
            vocab.add_token(&ch.to_string()).unwrap();
        }

        // Add a merge
        vocab.add_token("ab").unwrap();
        let token_id = vocab.get_id("ab").unwrap();

        let mut merges = MergeMap::new();
        merges.insert((0, 1), (0, token_id)); // Merge bytes 0 and 1

        let mut vocab_map = Vocab::new();
        let mut vocab_r_map = VocabR::new();
        for (k, v) in vocab.vocab.iter() {
            vocab_map.insert(k.clone(), *v);
        }
        for (k, v) in vocab.vocab_r.iter() {
            vocab_r_map.insert(*k, v.clone());
        }

        let encoder = ByteLevelEncoder::new(vocab_map, vocab_r_map, merges);

        // Test encoding
        let result = encoder.encode("\x00\x01").unwrap();
        assert_eq!(result, vec![token_id]);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut vocab = Vocabulary::new();

        // Add base byte vocabulary
        for i in 0..256u32 {
            let ch = unsafe { char::from_u32_unchecked(256 + i) };
            vocab.add_token(&ch.to_string()).unwrap();
        }

        let mut vocab_map = Vocab::new();
        let mut vocab_r_map = VocabR::new();
        for (k, v) in vocab.vocab.iter() {
            vocab_map.insert(k.clone(), *v);
        }
        for (k, v) in vocab.vocab_r.iter() {
            vocab_r_map.insert(*k, v.clone());
        }

        let merges = MergeMap::new();
        let encoder = ByteLevelEncoder::new(vocab_map, vocab_r_map, merges);

        let text = "Hello, world!";
        let encoded = encoder.encode(text).unwrap();
        let decoded = encoder.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }
}
