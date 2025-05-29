use cl::codegen::settings::Configurable as _;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods},
    Bound, PyResult,
};

#[pyclass]
pub struct TargetIsa {
    pub(crate) isa: cl::codegen::isa::OwnedTargetIsa,
}

#[pymethods]
impl TargetIsa {
    #[new]
    #[pyo3(signature = (triple, **kwds))]
    pub fn new(triple: &str, kwds: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let inner = match triple {
            "native" => cl::native::builder().map_err(|e| {
                PyValueError::new_err(format!("Failed to create native ISA: {}", e))
            })?,
            _ => cl::codegen::isa::lookup_by_name(triple).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to lookup target ISA for triple '{}': {}",
                    triple, e
                ))
            })?,
        };

        let mut flags = cl::codegen::settings::builder();

        if let Some(kwds) = kwds {
            for (key, value) in kwds.as_borrowed().iter() {
                let key_str = key.extract::<&str>().map_err(|e| {
                    PyTypeError::new_err(format!("Key must be a string, got: {}", e))
                })?;

                let value_str = value.extract::<&str>().map_err(|e| {
                    PyTypeError::new_err(format!(
                        "Value for '{}' must be a string, got: {}",
                        key_str, e
                    ))
                })?;

                flags.set(key_str, value_str).map_err(|e| {
                    PyValueError::new_err(format!("Failed to set flag '{}': {}", key_str, e))
                })?;
            }
        }

        let isa = inner
            .finish(cl::codegen::settings::Flags::new(flags))
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to create target ISA for triple '{}': {}",
                    triple, e
                ))
            })?;

        Ok(Self { isa })
    }

    pub fn __repr__(&self) -> String {
        format!("TargetIsa(\"{}\")", self.isa.triple())
    }
}
