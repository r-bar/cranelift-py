use pyo3::prelude::*;

pub mod codegen;
pub mod entities;
pub mod jit;
pub mod object;

#[pymodule]
pub mod cranelift {
    use super::*;

    #[pymodule_export]
    pub use crate::codegen::isa::TargetIsa;

    #[pymodule_export]
    pub use crate::codegen::ir::{MemFlags, Signature};

    #[pymodule_export]
    pub use crate::codegen::FunctionBuilder;

    #[pymodule_export]
    use crate::entities::{
        Block, CallConv, Constant, DynamicStackSlot, FuncId, FuncRef, GlobalValue, Immediate, Inst,
        JumpTable, Linkage, SigRef, StackSlot, TrapCode, Type, Value, ValueLabel, Variable,
    };

    #[pymodule_export]
    use crate::object::{ObjectModule, ObjectProduct};

    #[pymodule_export]
    pub use crate::jit::JITModule;

    #[pymodule_init]
    fn init_cranelift(m: &Bound<'_, PyModule>) -> PyResult<()> {
        TrapCode::init_class(m.getattr("TrapCode")?)?;
        Ok(())
    }
}
