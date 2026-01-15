//! Python wrapper for EncodingMode enum

use beepe_tokenizer::EncodingMode;
use pyo3::prelude::*;

/// Python wrapper for EncodingMode
#[pyclass(name = "EncodingMode")]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct PyEncodingMode {
    pub(crate) inner: EncodingMode,
}

#[pymethods]
impl PyEncodingMode {
    /// Byte-level encoding (tiktoken-style)
    #[staticmethod]
    fn byte_level() -> Self {
        Self {
            inner: EncodingMode::ByteLevel,
        }
    }

    /// Character/grapheme-level encoding
    #[staticmethod]
    fn char_level() -> Self {
        Self {
            inner: EncodingMode::CharLevel,
        }
    }

    /// Hybrid mode
    #[staticmethod]
    fn hybrid() -> Self {
        Self {
            inner: EncodingMode::Hybrid,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            EncodingMode::ByteLevel => "EncodingMode.ByteLevel".to_string(),
            EncodingMode::CharLevel => "EncodingMode.CharLevel".to_string(),
            EncodingMode::Hybrid => "EncodingMode.Hybrid".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self.inner {
            EncodingMode::ByteLevel => "byte_level".to_string(),
            EncodingMode::CharLevel => "char_level".to_string(),
            EncodingMode::Hybrid => "hybrid".to_string(),
        }
    }

    /// Hash support for using in sets/dicts
    fn __hash__(&self) -> u64 {
        match self.inner {
            EncodingMode::ByteLevel => 0,
            EncodingMode::CharLevel => 1,
            EncodingMode::Hybrid => 2,
        }
    }

    /// Equality comparison
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl From<PyEncodingMode> for EncodingMode {
    fn from(py_mode: PyEncodingMode) -> Self {
        py_mode.inner
    }
}
