//! PyO3 wrapper for the Tokenizer struct

use beepe_tokenizer::{Encoding, Tokenizer, TokenizerBuilder};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::PathBuf;

use crate::encoding_mode::PyEncodingMode;
use crate::special_tokens::PySpecialTokensConfig;

// Import the trait for error conversion
use crate::error::IntoPyErr;

/// Python wrapper for the beepe Tokenizer
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a directory
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self>{
        let path_buf = PathBuf::from(path);
        let inner = Tokenizer::load(&path_buf).map_err(|e| e.into_py_err())?;
        Ok(PyTokenizer { inner })
    }

    /// Load a tokenizer from HuggingFace format
    #[staticmethod]
    fn load_huggingface(path: &str) -> PyResult<Self>{
        let path_buf = PathBuf::from(path);
        let inner = Tokenizer::load_huggingface(&path_buf).map_err(|e| e.into_py_err())?;
        Ok(PyTokenizer { inner })
    }

    /// Create a new tokenizer with default configuration
    #[staticmethod]
    fn builder() -> PyTokenizerBuilder {
        PyTokenizerBuilder::new()
    }

    /// Encode text to token IDs
    #[pyo3(signature = (text, add_special_tokens=false))]
    fn encode(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>>{
        let encoding: Encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| e.into_py_err())?;
        Ok(encoding.ids)
    }

    /// Encode a batch of texts
    #[pyo3(signature = (texts, add_special_tokens=false))]
    fn encode_batch(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>>{
        let encodings: Vec<Encoding> = self
            .inner
            .encode_batch(&texts, add_special_tokens)
            .map_err(|e| e.into_py_err())?;
        Ok(encodings.into_iter().map(|e| e.ids).collect())
    }

    /// Decode token IDs back to text
    #[pyo3(signature = (ids, skip_special_tokens=false))]
    fn decode(&self, ids: &Bound<'_, PyList>, skip_special_tokens: bool) -> PyResult<String>{
        // Convert Python list to Vec<u32>
        let rust_ids: Vec<u32> = ids
            .iter()
            .map(|item| {
                item.extract::<u32>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Token IDs must be integers")
                })
            })
            .collect::<PyResult<Vec<u32>>>()?;

        Ok(self.inner.decode(&rust_ids, skip_special_tokens))
    }

    /// Get the vocabulary size
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Save the tokenizer to a directory
    fn save(&self, path: &str) -> PyResult<()>{
        let path_buf = PathBuf::from(path);
        self.inner.save(&path_buf).map_err(|e| e.into_py_err())?;
        Ok(())
    }

    /// Train the tokenizer on text data
    fn train(&mut self, data: &str) -> PyResult<()>{
        self.inner.train(data).map_err(|e| e.into_py_err())?;
        Ok(())
    }

    /// Get a string representation
    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }

    /// Get a string representation
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Encode text without special tokens
    fn encode_ordinary(&self, text: &str) -> PyResult<Vec<u32>>{
        let encoding: Encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| e.into_py_err())?;
        Ok(encoding.ids)
    }

    /// Encode a single token from bytes
    fn encode_single_token(&self, text: &[u8]) -> PyResult<u32>{
        let token_str = std::str::from_utf8(text)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid UTF-8 bytes"))?;

        let token_id = self.inner.vocab().get_id(token_str).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Token not found: {}", token_str))
        })?;

        Ok(token_id)
    }

    /// Decode single token to bytes
    fn decode_single_token_bytes(&self, token_id: u32) -> PyResult<Vec<u8>>{
        let token_str = self.inner.vocab().get_token(token_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Token ID not found: {}",
                token_id
            ))
        })?;

        Ok(token_str.as_bytes().to_vec())
    }

    /// Encode batch without special tokens
    fn encode_ordinary_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>>{
        let encodings: Vec<Encoding> = self
            .inner
            .encode_batch(&texts, false)
            .map_err(|e| e.into_py_err())?;
        Ok(encodings.into_iter().map(|e| e.ids).collect())
    }

    /// Decode batch of token lists
    fn decode_batch(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>>{
        let results: Vec<String> = token_lists
            .into_iter()
            .map(|ids| self.inner.decode(&ids, false))
            .collect();
        Ok(results)
    }

    /// Get vocabulary size (alias for vocab_size)
    fn n_vocab(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get maximum valid token ID
    fn max_token_value(&self) -> usize {
        self.inner.vocab_size() - 1
    }

    /// Get encoding name
    fn name(&self) -> String {
        "beepe_byte_level".to_string()
    }

    /// Decode token IDs back to text with character offsets
    fn decode_with_offsets(&self, ids: &Bound<'_, PyList>) -> PyResult<(String, Vec<usize>)>{
        let rust_ids: Vec<u32> = ids
            .iter()
            .map(|item| item.extract::<u32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Token IDs must be integers")
            })?;

        let text = self.inner.decode(&rust_ids, false);

        // Calculate character offsets by incrementally decoding and counting
        let mut offsets: Vec<usize> = Vec::with_capacity(rust_ids.len());
        let mut char_pos = 0;

        for i in 0..rust_ids.len() {
            offsets.push(char_pos);

            // Decode token i and count its contribution
            let token_decoded = self.inner.decode(&rust_ids[..=i], false);

            // If we got a decoding error placeholder, count as 1 char to progress
            // Otherwise count actual characters (up to remaining text length)
            let new_char_pos = if token_decoded.starts_with("<") && token_decoded.ends_with(">") {
                char_pos + 1
            } else {
                token_decoded.chars().count().min((i + 1) * 10) // Cap at reasonable max
            };
            char_pos = new_char_pos.min(text.chars().count().saturating_sub(1).max(char_pos));
        }

        // Ensure all offsets are within text character bounds
        let text_chars = text.chars().count();
        let offsets: Vec<usize> = offsets
            .into_iter()
            .map(|o| o.min(text_chars.saturating_sub(1).max(0)))
            .collect();

        Ok((text, offsets))
    }
}

/// Builder for creating tokenizers with custom configuration
#[pyclass(name = "TokenizerBuilder")]
pub struct PyTokenizerBuilder {
    inner: TokenizerBuilder,
}

#[pymethods]
impl PyTokenizerBuilder {
    #[staticmethod]
    fn new() -> Self {
        Self {
            inner: Tokenizer::builder(),
        }
    }

    fn vocab_size(&self, size: usize) -> Self {
        Self {
            inner: self.inner.clone().vocab_size(size),
        }
    }

    fn min_frequency(&self, freq: u64) -> Self {
        Self {
            inner: self.inner.clone().min_frequency(freq),
        }
    }

    fn encoding_mode(&self, mode: &PyEncodingMode) -> Self {
        Self {
            inner: self.inner.clone().encoding_mode(mode.inner),
        }
    }

    fn with_special_tokens(&self, tokens: &PySpecialTokensConfig) -> Self {
        Self {
            inner: self.inner.clone().with_special_tokens(tokens.inner.clone()),
        }
    }

    fn build(&self) -> PyResult<PyTokenizer>{
        let inner = self.inner.clone().build().map_err(|e| e.into_py_err())?;
        Ok(PyTokenizer { inner })
    }
}
