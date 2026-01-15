//! Python wrapper for SpecialTokensConfig

use beepe_core::SpecialTokensConfig as RustSpecialTokensConfig;
use pyo3::prelude::*;

/// Python wrapper for SpecialTokensConfig
#[pyclass(name = "SpecialTokensConfig")]
#[derive(Clone, Default)]
pub struct PySpecialTokensConfig {
    pub(crate) inner: RustSpecialTokensConfig,
}

#[pymethods]
impl PySpecialTokensConfig {
    #[staticmethod]
    fn new() -> Self {
        Self::default()
    }

    #[getter]
    fn pad(&self) -> Option<String> {
        self.inner.pad.clone()
    }

    #[setter]
    fn set_pad(&mut self, value: Option<String>) {
        self.inner.pad = value;
    }

    #[getter]
    fn unk(&self) -> Option<String> {
        self.inner.unk.clone()
    }

    #[setter]
    fn set_unk(&mut self, value: Option<String>) {
        self.inner.unk = value;
    }

    #[getter]
    fn bos(&self) -> Option<String> {
        self.inner.bos.clone()
    }

    #[setter]
    fn set_bos(&mut self, value: Option<String>) {
        self.inner.bos = value;
    }

    #[getter]
    fn eos(&self) -> Option<String> {
        self.inner.eos.clone()
    }

    #[setter]
    fn set_eos(&mut self, value: Option<String>) {
        self.inner.eos = value;
    }

    #[getter]
    fn mask(&self) -> Option<String> {
        self.inner.mask.clone()
    }

    #[setter]
    fn set_mask(&mut self, value: Option<String>) {
        self.inner.mask = value;
    }

    #[getter]
    fn user(&self) -> Option<String> {
        self.inner.user.clone()
    }

    #[setter]
    fn set_user(&mut self, value: Option<String>) {
        self.inner.user = value;
    }

    #[getter]
    fn assistant(&self) -> Option<String> {
        self.inner.assistant.clone()
    }

    #[setter]
    fn set_assistant(&mut self, value: Option<String>) {
        self.inner.assistant = value;
    }

    #[getter]
    fn system(&self) -> Option<String> {
        self.inner.system.clone()
    }

    #[setter]
    fn set_system(&mut self, value: Option<String>) {
        self.inner.system = value;
    }

    fn __repr__(&self) -> String {
        format!("SpecialTokensConfig({:?})", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<PySpecialTokensConfig> for RustSpecialTokensConfig {
    fn from(py_config: PySpecialTokensConfig) -> Self {
        py_config.inner
    }
}

impl From<RustSpecialTokensConfig> for PySpecialTokensConfig {
    fn from(rust_config: RustSpecialTokensConfig) -> Self {
        Self { inner: rust_config }
    }
}
