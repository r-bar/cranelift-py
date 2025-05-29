use cl::prelude::AbiParam;
use pyo3::{pyclass, pymethods, PyResult};

use crate::entities::{AliasRegion, CallConv, Endianness, TrapCode, Type};

#[pyclass]
#[derive(Clone, Debug)]
pub struct Signature {
    pub(crate) signature: cl::codegen::ir::Signature,
}

#[pymethods]
impl Signature {
    #[new]
    pub fn new(call_conv: &CallConv) -> Self {
        Self {
            signature: cl::codegen::ir::Signature::new((*call_conv).into()),
        }
    }

    pub fn add_param(&mut self, ty: Type) {
        self.signature.params.push(AbiParam::new(ty.into()));
    }

    pub fn add_return(&mut self, ty: Type) {
        self.signature.returns.push(AbiParam::new(ty.into()));
    }

    #[getter]
    pub fn params(&self) -> Vec<Type> {
        self.signature
            .params
            .iter()
            .map(|param| Type::from(param.value_type))
            .collect()
    }

    #[getter]
    pub fn returns(&self) -> Vec<Type> {
        self.signature
            .returns
            .iter()
            .map(|param| Type::from(param.value_type))
            .collect()
    }
}

impl AsRef<cl::codegen::ir::Signature> for Signature {
    fn as_ref(&self) -> &cl::codegen::ir::Signature {
        &self.signature
    }
}

impl From<Signature> for cl::codegen::ir::Signature {
    fn from(sig: Signature) -> Self {
        sig.signature
    }
}

impl From<cl::codegen::ir::Signature> for Signature {
    fn from(signature: cl::codegen::ir::Signature) -> Self {
        Signature { signature }
    }
}

#[pyclass(eq)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemFlags {
    flags: cl::codegen::ir::MemFlags,
}

#[pymethods]
impl MemFlags {
    #[new]
    pub fn new() -> Self {
        MemFlags {
            flags: cl::codegen::ir::MemFlags::new(),
        }
    }

    #[staticmethod]
    pub fn trusted() -> Self {
        MemFlags {
            flags: cl::codegen::ir::MemFlags::trusted(),
        }
    }

    pub fn alias_region(&self) -> Option<AliasRegion> {
        self.flags.alias_region().map(Into::into)
    }

    pub fn set_alias_region(&mut self, region: Option<AliasRegion>) {
        self.flags.set_alias_region(region.map(Into::into));
    }

    pub fn set_by_name(&mut self, name: &str) -> PyResult<()> {
        self.flags.set_by_name(name).map(|_| ()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to set memory flags by name '{}': {}",
                name, e
            ))
        })
    }

    pub fn endianness(&self, native_endianness: Endianness) -> Endianness {
        self.flags.endianness(native_endianness.into()).into()
    }

    pub fn explicit_endianness(&self) -> Option<Endianness> {
        self.flags.explicit_endianness().map(Into::into)
    }

    pub fn set_endianness(&mut self, endianness: Endianness) {
        self.flags.set_endianness(endianness.into());
    }

    pub fn notrap(&self) -> bool {
        self.flags.notrap()
    }

    pub fn set_notrap(&mut self) {
        self.flags.set_notrap();
    }

    pub fn can_move(&self) -> bool {
        self.flags.can_move()
    }

    pub fn set_can_move(&mut self) {
        self.flags.set_can_move();
    }

    pub fn aligned(&self) -> bool {
        self.flags.aligned()
    }

    pub fn set_aligned(&mut self) {
        self.flags.set_aligned();
    }

    pub fn readonly(&self) -> bool {
        self.flags.readonly()
    }

    pub fn set_readonly(&mut self) {
        self.flags.set_readonly();
    }

    pub fn checked(&self) -> bool {
        self.flags.checked()
    }

    pub fn set_checked(&mut self) {
        self.flags.set_checked();
    }

    pub fn trap_code(&self) -> Option<TrapCode> {
        self.flags.trap_code().map(Into::into)
    }

    pub fn set_trap_code(&mut self, code: Option<TrapCode>) {
        self.flags = self.flags.with_trap_code(code.map(Into::into));
    }
}

impl From<cl::codegen::ir::MemFlags> for MemFlags {
    fn from(flags: cl::codegen::ir::MemFlags) -> Self {
        MemFlags { flags }
    }
}

impl From<MemFlags> for cl::codegen::ir::MemFlags {
    fn from(flags: MemFlags) -> Self {
        flags.flags
    }
}
