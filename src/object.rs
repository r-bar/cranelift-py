use cl::module::{default_libcall_names, Module as _};
use pyo3::{pyclass, pymethods, PyResult};

use crate::{
    codegen::{ir::Signature, FunctionBuilder},
    cranelift::TargetIsa,
    entities::{FuncId, Linkage},
};

#[pyclass]
pub struct ObjectModule {
    pub(crate) module: Option<cl::object::ObjectModule>,
}

#[pymethods]
impl ObjectModule {
    #[new]
    pub fn new(isa: &TargetIsa, name: &str) -> PyResult<Self> {
        let builder =
            cl::object::ObjectBuilder::new(isa.isa.clone(), name, default_libcall_names())
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to create ObjectBuilder: {}",
                        e
                    ))
                })?;

        let module: cl::object::ObjectModule = cl::object::ObjectModule::new(builder);

        Ok(ObjectModule {
            module: Some(module),
        })
    }

    pub fn declare_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        signature: &Signature,
    ) -> PyResult<FuncId> {
        self.module
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ObjectModule is finalized"))?
            .declare_function(name, linkage.into(), signature.as_ref())
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to declare function: {}",
                    e
                ))
            })
            .map(|id| id.into())
    }

    pub fn define_function(&mut self, func_id: FuncId, body: &mut FunctionBuilder) -> PyResult<()> {
        self.module
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ObjectModule is finalized"))?
            .define_function(func_id.into(), body.as_mut())
            .map_err(|e| {
                println!("{:?}", e);
                pyo3::exceptions::PyValueError::new_err(format!("Failed to define function: {}", e))
            })
    }

    pub fn finish(&mut self) -> PyResult<ObjectProduct> {
        Ok(self
            .module
            .take()
            .ok_or(pyo3::exceptions::PyValueError::new_err(
                "ObjectModule is already finalized",
            ))?
            .finish()
            .into())
    }
}

#[pyclass]
pub struct ObjectProduct {
    pub(crate) product: Option<cl::object::ObjectProduct>,
}

#[pymethods]
impl ObjectProduct {
    pub fn emit(&mut self) -> PyResult<Vec<u8>> {
        self.product
            .take()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("ObjectProduct is already emitted"))
            })?
            .emit()
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to emit object: {}", e))
            })
    }
}

impl From<cl::object::ObjectProduct> for ObjectProduct {
    fn from(product: cl::object::ObjectProduct) -> Self {
        ObjectProduct {
            product: Some(product),
        }
    }
}
