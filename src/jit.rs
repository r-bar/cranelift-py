use cl::module::{default_libcall_names, Module as _};
use pyo3::{pyclass, pymethods, PyResult};

use crate::{
    codegen::FunctionBuilder,
    cranelift::{Signature, TargetIsa},
    entities::{FuncId, Linkage},
};

#[pyclass]
pub struct JITModule {
    pub(crate) module: cl::jit::JITModule,
}

unsafe impl Send for JITModule {}
unsafe impl Sync for JITModule {}

#[pymethods]
impl JITModule {
    #[new]
    #[pyo3(signature = (isa=None))]
    pub fn new(isa: Option<&TargetIsa>) -> PyResult<Self> {
        let builder = match isa {
            Some(isa) => cl::jit::JITBuilder::with_isa(isa.isa.clone(), default_libcall_names()),
            None => cl::jit::JITBuilder::new(default_libcall_names()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to create JIT: {}", e))
            })?,
        };

        let module = cl::jit::JITModule::new(builder);

        Ok(JITModule { module })
    }

    pub fn declare_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        signature: &Signature,
    ) -> PyResult<FuncId> {
        self.module
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
            .define_function(func_id.into(), body.as_mut())
            .map_err(|e| {
                println!("{:?}", e);
                pyo3::exceptions::PyValueError::new_err(format!("Failed to define function: {}", e))
            })
    }

    pub fn finalize_definitions(&mut self) -> PyResult<()> {
        self.module.finalize_definitions().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to finalize definitions: {}",
                e
            ))
        })
    }

    pub fn get_finalized_function(&self, func_id: FuncId) -> usize {
        self.module.get_finalized_function(func_id.into()) as usize
    }
}
