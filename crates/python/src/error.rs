//! Error handling for Python bindings

use beepe_tokenizer::TokenizerError as RustTokenizerError;
use pyo3::{create_exception, exceptions::PyRuntimeError, PyErr};

/// Custom Python exception for tokenizer errors
create_exception!(
    beepe,
    TokenizerError,
    PyRuntimeError,
    "Error during tokenization"
);

/// Result type for tokenizer operations
pub type TokenizerResult<T> = Result<T, PyErr>;

/// Convert a Rust TokenizerError to a Python exception
pub trait IntoPyErr {
    fn into_py_err(self) -> PyErr;
}

impl IntoPyErr for RustTokenizerError {
    fn into_py_err(self) -> PyErr {
        TokenizerError::new_err(self.to_string())
    }
}
