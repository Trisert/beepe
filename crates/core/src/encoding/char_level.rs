//! Character-level BPE encoding.
//!
//! This module implements character/grapheme-level BPE encoding,
//! which treats each grapheme cluster as a token.

use crate::core::{vocab::Vocab, MergeMap};
use crate::Result;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use unicode_segmentation::UnicodeSegmentation;

/// Trie node for efficient longest-match tokenization.
#[derive(Debug, Clone)]
struct CharTrieNode {
    /// Child nodes indexed by grapheme cluster
    children: AHashMap<String, CharTrieNode>,
    /// Token ID if this node represents a complete token
    token_id: Option<u32>,
}

impl CharTrieNode {
    fn new() -> Self {
        Self {
            children: AHashMap::new(),
            token_id: None,
        }
    }
}

/// Trie structure for fast prefix lookup of tokens at grapheme level.
struct CharVocabTrie {
    root: CharTrieNode,
}

impl CharVocabTrie {
    fn new() -> Self {
        Self {
            root: CharTrieNode::new(),
        }
    }

    /// Insert a token string into the trie.
    fn insert(&mut self, token: &str, token_id: u32) {
        let mut node = &mut self.root;

        // Split token into grapheme clusters
        for grapheme in token.graphemes(true) {
            node = node
                .children
                .entry(grapheme.to_string())
                .or_insert_with(CharTrieNode::new);
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
    fn find_longest_match(&self, graphemes: &[&str], pos: usize) -> Option<(u32, usize)> {
        let mut node = &self.root;
        let mut best_match: Option<(u32, usize)> = None;

        for i in pos..graphemes.len() {
            let g = graphemes[i];

            match node.children.get(g) {
                Some(child) => {
                    node = child;

                    if let Some(token_id) = node.token_id {
                        best_match = Some((token_id, i - pos + 1));
                    }
                }
                None => {
                    break;
                }
            }
        }

        best_match
    }
}

/// Character-level BPE encoder.
///
/// This encoder treats each grapheme cluster as a token and applies
/// BPE merges at the character level.
pub struct CharLevelEncoder {
    /// Vocabulary for token lookups
    vocab: Vocab,
    /// Reverse vocabulary for decoding
    vocab_r: crate::core::vocab::VocabR,
    /// Merge rules: pair -> (rank, new_token_id)
    merges: MergeMap,
    /// Trie for efficient longest-match tokenization
    vocab_trie: CharVocabTrie,
}

impl CharLevelEncoder {
    /// Create a new character-level encoder.
    pub fn new(vocab: Vocab, vocab_r: crate::core::vocab::VocabR, merges: MergeMap) -> Self {
        let vocab_trie = CharVocabTrie::from_vocab(&vocab);

        Self {
            vocab,
            vocab_r,
            merges,
            vocab_trie,
        }
    }

    /// Encode text using character-level BPE with longest-match tokenization.
    ///
    /// Returns a vector of token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Split into grapheme clusters
        let graphemes: Vec<&str> = text.graphemes(true).collect();

        // Longest-match tokenization using trie (O(n * max_token_length) instead of O(n²))
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < graphemes.len() {
            // Use trie to find the longest matching token at this position
            match self.vocab_trie.find_longest_match(&graphemes, pos) {
                Some((token_id, length)) => {
                    tokens.push(token_id);
                    pos += length;
                }
                None => {
                    // No match found - fallback to single grapheme
                    let g = graphemes[pos];
                    let token_id = self
                        .vocab
                        .get(g)
                        .ok_or_else(|| crate::TokenizerError::UnknownToken(g.to_string()))?;
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

        impl Ord for MergeOp {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.rank
                    .cmp(&other.rank)
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
            if merge_op.pos + 1 >= tokens.len() {
                continue;
            }

            let pair = (tokens[merge_op.pos], tokens[merge_op.pos + 1]);

            if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                if rank != merge_op.rank || new_id != merge_op.new_id {
                    continue;
                }

                tokens[merge_op.pos] = merge_op.new_id;
                tokens.remove(merge_op.pos + 1);

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

                if merge_op.pos < tokens.len() {
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
        let tokens: Vec<String> = ids
            .iter()
            .map(|&id| {
                self.vocab_r
                    .get(&id)
                    .ok_or(crate::TokenizerError::UnknownTokenId(id))
                    .map(|s| s.to_string())
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(tokens.join(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vocab::Vocabulary;

    #[test]
    fn test_encode_simple() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("h").unwrap();
        vocab.add_token("e").unwrap();
        vocab.add_token("l").unwrap();
        vocab.add_token("o").unwrap();
        vocab.add_token(" ").unwrap();

        let mut vocab_map = Vocab::new();
        let mut vocab_r_map = crate::core::vocab::VocabR::new();
        for (k, v) in vocab.vocab.iter() {
            vocab_map.insert(k.clone(), *v);
        }
        for (k, v) in vocab.vocab_r.iter() {
            vocab_r_map.insert(*k, v.clone());
        }

        let merges = crate::core::MergeMap::new();
        let encoder = CharLevelEncoder::new(vocab_map, vocab_r_map, merges);

        let result = encoder.encode("hello").unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut vocab = Vocabulary::new();

        // Add some characters
        for c in ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd', '!'] {
            vocab.add_token(&c.to_string()).unwrap();
        }

        let mut vocab_map = Vocab::new();
        let mut vocab_r_map = crate::core::vocab::VocabR::new();
        for (k, v) in vocab.vocab.iter() {
            vocab_map.insert(k.clone(), *v);
        }
        for (k, v) in vocab.vocab_r.iter() {
            vocab_r_map.insert(*k, v.clone());
        }

        let merges = crate::core::MergeMap::new();
        let encoder = CharLevelEncoder::new(vocab_map, vocab_r_map, merges);

        let text = "hello world!";
        let encoded = encoder.encode(text).unwrap();
        let decoded = encoder.decode(&encoded).unwrap();

        assert_eq!(decoded, text);
    }
}
