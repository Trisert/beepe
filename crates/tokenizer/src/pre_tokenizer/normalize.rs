//! Unicode normalization for pre-tokenization.
//!
//! This module provides Unicode normalization operations (NFC, NFD, NFKC, NFKD)
//! that are commonly applied before BPE encoding.

use unicode_normalization::UnicodeNormalization;

/// Normalization form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizationForm {
    /// Canonical composition
    #[default]
    NFC,
    /// Canonical decomposition
    NFD,
    /// Compatibility composition
    NFKC,
    /// Compatibility decomposition
    NFKD,
    /// No normalization
    None,
}

/// Unicode normalizer.
pub struct Normalizer {
    /// Normalization form to apply
    form: NormalizationForm,
}

impl Normalizer {
    /// Create a new normalizer.
    pub fn new(form: NormalizationForm) -> Self {
        Self { form }
    }

    /// Create an NFC normalizer (default).
    pub fn nfc() -> Self {
        Self::new(NormalizationForm::NFC)
    }

    /// Normalize text.
    pub fn normalize(&self, text: &str) -> String {
        match self.form {
            NormalizationForm::NFC => text.nfc().collect(),
            NormalizationForm::NFD => text.nfd().collect(),
            NormalizationForm::NFKC => text.nfkc().collect(),
            NormalizationForm::NFKD => text.nfkd().collect(),
            NormalizationForm::None => text.to_string(),
        }
    }

    /// Check if normalization is enabled.
    pub fn is_enabled(&self) -> bool {
        self.form != NormalizationForm::None
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::nfc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfc_normalization() {
        let normalizer = Normalizer::nfc();
        // Combining characters
        let text = "e\u{0301}"; // e + combining acute accent
        let result = normalizer.normalize(text);
        assert_eq!(result, "\u{00e9}"); // é as single character
    }

    #[test]
    fn test_nfd_normalization() {
        let normalizer = Normalizer::new(NormalizationForm::NFD);
        let text = "\u{00e9}"; // é as single character
        let result = normalizer.normalize(text);
        assert_eq!(result, "e\u{0301}"); // é decomposed
    }

    #[test]
    fn test_no_normalization() {
        let normalizer = Normalizer::new(NormalizationForm::None);
        let text = "Hello";
        let result = normalizer.normalize(text);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_is_enabled() {
        let nfc = Normalizer::nfc();
        assert!(nfc.is_enabled());

        let none = Normalizer::new(NormalizationForm::None);
        assert!(!none.is_enabled());
    }
}
