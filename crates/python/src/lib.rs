//! Python bindings for beepe tokenizer
//!
//! This module provides a Pythonic interface to the Rust-based beepe tokenizer.

use pyo3::prelude::*;

mod encoding_mode;
mod error;
mod special_tokens;
mod tokenizer;

use encoding_mode::PyEncodingMode;
use special_tokens::PySpecialTokensConfig;
use tokenizer::PyTokenizer;

/// beepe: Fast BPE tokenizer in Rust with Python bindings
#[pymodule]
fn beepe(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyEncodingMode>()?;
    m.add_class::<PySpecialTokensConfig>()?;
    m.add_class::<tokenizer::PyTokenizerBuilder>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
