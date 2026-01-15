//! Text splitting for pre-tokenization.
//!
//! This module provides various text splitting strategies used before
//! BPE encoding, such as whitespace splitting and regex-based splitting.

use regex::Regex;
use std::sync::OnceLock;

/// Text splitter for pre-tokenization.
pub struct Splitter {
    /// Pattern to split on
    pattern: SplitPattern,
}

/// Splitting patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitPattern {
    /// No splitting (keep text as-is)
    NoSplit,
    /// Split on whitespace
    Whitespace,
    /// Split on any character (character-level)
    Character,
    /// Custom regex pattern
    Custom(&'static str),
}

impl Splitter {
    /// Create a new splitter.
    pub fn new(pattern: SplitPattern) -> Self {
        Self { pattern }
    }

    /// Create a whitespace splitter.
    pub fn whitespace() -> Self {
        Self::new(SplitPattern::Whitespace)
    }

    /// Create a character-level splitter.
    pub fn character() -> Self {
        Self::new(SplitPattern::Character)
    }

    /// Split text into chunks.
    pub fn split(&self, text: &str) -> Vec<String> {
        match self.pattern {
            SplitPattern::NoSplit => vec![text.to_string()],
            SplitPattern::Whitespace => text.split_whitespace().map(|s| s.to_string()).collect(),
            SplitPattern::Character => text.chars().map(|c| c.to_string()).collect(),
            SplitPattern::Custom(pattern) => {
                static RE: OnceLock<Regex> = OnceLock::new();
                let re = RE.get_or_init(|| Regex::new(pattern).expect("Invalid regex pattern"));
                re.split(text).map(|s| s.to_string()).collect()
            }
        }
    }
}

impl Default for Splitter {
    fn default() -> Self {
        Self::new(SplitPattern::NoSplit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_split() {
        let splitter = Splitter::whitespace();
        let result = splitter.split("hello world  test");
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_nosplit() {
        let splitter = Splitter::new(SplitPattern::NoSplit);
        let result = splitter.split("hello world  test");
        assert_eq!(result, vec!["hello world  test"]);
    }

    #[test]
    fn test_character_split() {
        let splitter = Splitter::character();
        let result = splitter.split("hi");
        assert_eq!(result, vec!["h", "i"]);
    }

    #[test]
    fn test_custom_split() {
        let splitter = Splitter::new(SplitPattern::Custom(r"\s+"));
        let result = splitter.split("hello  world");
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn test_empty_string() {
        let splitter = Splitter::whitespace();
        let result = splitter.split("");
        assert_eq!(result, Vec::<String>::new());
    }
}
