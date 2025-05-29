use pyo3::{pyclass, pymethods, types::PyAnyMethods, Bound, PyAny, PyResult};

use cl::codegen::{entity::EntityRef as _, ir::condcodes::CondCode as _};

use crate::cranelift::TargetIsa;

macro_rules! entity_wrap {
    ($name:ident, $target:ty) => {
        #[pyclass(eq, hash, frozen)]
        #[derive(PartialEq, Debug, Clone, Copy)]
        pub struct $name($target);

        impl From<$target> for $name {
            fn from(value: $target) -> Self {
                $name(value)
            }
        }

        impl From<$name> for $target {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.index().hash(state);
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(index: usize) -> Self {
                $name(<$target>::new(index))
            }

            pub fn __repr__(&self) -> String {
                format!("{}({})", stringify!($name), self.0.index())
            }

            pub fn __str__(&self) -> String {
                format!("{}", self.0)
            }

            #[getter]
            pub fn index(&self) -> usize {
                self.0.index()
            }
        }
    };
}

entity_wrap!(FuncRef, cl::codegen::ir::FuncRef);
entity_wrap!(SigRef, cl::codegen::ir::SigRef);
entity_wrap!(Value, cl::codegen::ir::Value);
entity_wrap!(ValueLabel, cl::codegen::ir::ValueLabel);
entity_wrap!(Block, cl::codegen::ir::Block);
entity_wrap!(Variable, cl::frontend::Variable);
entity_wrap!(JumpTable, cl::codegen::ir::JumpTable);
entity_wrap!(Inst, cl::codegen::ir::Inst);
entity_wrap!(Constant, cl::codegen::ir::Constant);
entity_wrap!(Immediate, cl::codegen::ir::Immediate);
entity_wrap!(StackSlot, cl::codegen::ir::StackSlot);
entity_wrap!(DynamicStackSlot, cl::codegen::ir::DynamicStackSlot);
entity_wrap!(DynamicType, cl::codegen::ir::DynamicType);
entity_wrap!(UserExternalNameRef, cl::codegen::ir::UserExternalNameRef);
entity_wrap!(GlobalValue, cl::codegen::ir::GlobalValue);
entity_wrap!(FuncId, cl::module::FuncId);

impl From<Value> for cl::codegen::ir::BlockArg {
    fn from(value: Value) -> Self {
        cl::codegen::ir::BlockArg::Value(value.0)
    }
}

macro_rules! enum_wrap {
    ($name:ident, $target:ty, { $( $(#[ $attr:meta ])* $variant:ident $( = $value:expr )? , )* } $({ $($rest:tt)* })*) => {
        #[pyclass(eq, eq_int, hash, frozen)]
        #[derive(PartialEq, Debug, Clone, Copy, Hash)]
        pub enum $name {
            $(
                $(#[$attr])*
                $variant $( = $value )? ,
            )*
        }

        impl From<$name> for $target {
            fn from(value: $name) -> Self {
                match value {
                    $( $name::$variant $( => $value )? => <$target>::$variant, )*
                }
            }
        }

        impl From<$target> for $name {
            fn from(value: $target) -> Self {
                match value {
                    $( <$target>::$variant => $name::$variant, )*
                }
            }
        }

        #[pymethods]
        impl $name {
            pub fn __repr__(&self) -> String {
                format!("{}::{:?}", stringify!($name), self)
            }

            pub fn __str__(&self) -> String {
                format!("{:?}", self)
            }

            $($($rest)*)*
        }
    };
}

enum_wrap!(Linkage, cl::module::Linkage, {
    /// Defined outside of a module.
    Import,
    /// Defined inside the module, but not visible outside it.
    Local,
    /// Defined inside the module, visible outside it, and may be preempted.
    Preemptible,
    /// Defined inside the module, visible inside the current static linkage unit, but not outside.
    ///
    /// A static linkage unit is the combination of all object files passed to a linker to create
    /// an executable or dynamic library.
    Hidden,
    /// Defined inside the module, and visible outside it.
    Export,
});

enum_wrap!(CallConv, cl::codegen::isa::CallConv, {
    /// Best performance, not ABI-stable.
    Fast,
    /// Smallest caller code size, not ABI-stable.
    Cold,
    /// Supports tail calls, not ABI-stable except for exception
    /// payload registers.
    ///
    /// On exception resume, a caller to a `tail`-convention function
    /// assumes that the exception payload values are in the following
    /// registers (per platform):
    /// - x86-64: rax, rdx
    /// - aarch64: x0, x1
    /// - riscv64: a0, a1
    /// - pulley{32,64}: x0, x1
    //
    // Currently, this is basically sys-v except that callees pop stack
    // arguments, rather than callers. Expected to change even more in the
    // future, however!
    Tail,
    /// System V-style convention used on many platforms.
    SystemV,
    /// Windows "fastcall" convention, also used for x64 and ARM.
    WindowsFastcall,
    /// Mac aarch64 calling convention, which is a tweaked aarch64 ABI.
    AppleAarch64,
    /// Specialized convention for the probestack function.
    Probestack,
    /// The winch calling convention, not ABI-stable.
    ///
    /// The main difference to SystemV is that the winch calling convention
    /// defines no callee-save registers, and restricts the number of return
    /// registers to one integer, and one floating point.
    Winch,
});

macro_rules! int_wrap {
    ($name:ident, $target:ty, $repr:ty, { $($variant:ident = $value:expr , )* }) => {
        #[pyclass(eq, eq_int, hash, frozen)]
        #[derive(PartialEq, Debug, Clone, Copy, Hash)]
        #[repr($repr)]
        #[allow(non_camel_case_types)]
        pub enum $name {
            $(
                $variant = $value,
            )*
        }

        impl From<$name> for $target {
            fn from(value: $name) -> Self {
                unsafe { std::mem::transmute(value) }
            }
        }

        impl From<$target> for $name {
            fn from(value: $target) -> Self {
                unsafe { std::mem::transmute(value) }
            }
        }
    };
}

int_wrap!(Type, cl::codegen::ir::types::Type, u16, {
    INVALID = 0x00,
    I8 = 0x74,
    I16 = 0x75,
    I32 = 0x76,
    I64 = 0x77,
    I128 = 0x78,
    F16 = 0x79,
    F32 = 0x7a,
    F64 = 0x7b,
    F128 = 0x7c,
    I8X2 = 0x84,
    I8X2XN = 0x104,
    I8X4 = 0x94,
    I16X2 = 0x85,
    F16X2 = 0x89,
    I8X4XN = 0x114,
    I16X2XN = 0x105,
    F16X2XN = 0x109,
    I8X8 = 0xa4,
    I16X4 = 0x95,
    I32X2 = 0x86,
    F16X4 = 0x99,
    F32X2 = 0x8a,
    I8X8XN = 0x124,
    I16X4XN = 0x115,
    I32X2XN = 0x106,
    F16X4XN = 0x119,
    F32X2XN = 0x10a,
    I8X16 = 0xb4,
    I16X8 = 0xa5,
    I32X4 = 0x96,
    I64X2 = 0x87,
    F16X8 = 0xa9,
    F32X4 = 0x9a,
    F64X2 = 0x8b,
    I8X16XN = 0x134,
    I16X8XN = 0x125,
    I32X4XN = 0x116,
    I64X2XN = 0x107,
    F16X8XN = 0x129,
    F32X4XN = 0x11a,
    F64X2XN = 0x10b,
    I8X32 = 0xc4,
    I16X16 = 0xb5,
    I32X8 = 0xa6,
    I64X4 = 0x97,
    I128X2 = 0x88,
    F16X16 = 0xb9,
    F32X8 = 0xaa,
    F64X4 = 0x9b,
    F128X2 = 0x8c,
    I8X32XN = 0x144,
    I16X16XN = 0x135,
    I32X8XN = 0x126,
    I64X4XN = 0x117,
    I128X2XN = 0x108,
    F16X16XN = 0x139,
    F32X8XN = 0x12a,
    F64X4XN = 0x11b,
    F128X2XN = 0x10c,
    I8X64 = 0xd4,
    I16X32 = 0xc5,
    I32X16 = 0xb6,
    I64X8 = 0xa7,
    I128X4 = 0x98,
    F16X32 = 0xc9,
    F32X16 = 0xba,
    F64X8 = 0xab,
    F128X4 = 0x9c,
    I8X64XN = 0x154,
    I16X32XN = 0x145,
    I32X16XN = 0x136,
    I64X8XN = 0x127,
    I128X4XN = 0x118,
    F16X32XN = 0x149,
    F32X16XN = 0x13a,
    F64X8XN = 0x12b,
    F128X4XN = 0x11c,
});

#[pymethods]
impl Type {
    pub fn __repr__(&self) -> String {
        format!("Type::{:?}", self)
    }

    pub fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    pub fn lane_type(&self) -> Self {
        self.as_type().lane_type().into()
    }

    pub fn lane_of(&self) -> Self {
        self.lane_type()
    }

    pub fn log2_lane_bits(&self) -> u32 {
        self.as_type().log2_lane_bits()
    }

    pub fn lane_bits(&self) -> u32 {
        self.as_type().lane_bits()
    }

    pub fn bounds(&self, signed: bool) -> (u128, u128) {
        self.as_type().bounds(signed)
    }

    #[staticmethod]
    pub fn int(bits: u16) -> Option<Self> {
        cl::codegen::ir::types::Type::int(bits).map(Into::into)
    }

    #[staticmethod]
    pub fn int_with_byte_size(bytes: u16) -> Option<Self> {
        cl::codegen::ir::types::Type::int_with_byte_size(bytes).map(Into::into)
    }

    pub fn as_truthy_pedantic(&self) -> Self {
        self.as_type().as_truthy_pedantic().into()
    }

    pub fn as_truthy(&self) -> Self {
        self.as_type().as_truthy().into()
    }

    pub fn as_int(&self) -> Self {
        self.as_type().as_int().into()
    }

    pub fn half_width(&self) -> Option<Self> {
        self.as_type().half_width().map(Into::into)
    }

    pub fn double_width(&self) -> Option<Self> {
        self.as_type().double_width().map(Into::into)
    }

    pub fn is_invalid(&self) -> bool {
        self.as_type().is_invalid()
    }

    pub fn is_special(&self) -> bool {
        self.as_type().is_special()
    }

    pub fn is_lane(&self) -> bool {
        self.as_type().is_lane()
    }

    pub fn is_vector(&self) -> bool {
        self.as_type().is_vector()
    }

    pub fn is_dynamic_vector(&self) -> bool {
        self.as_type().is_dynamic_vector()
    }

    pub fn is_int(&self) -> bool {
        self.as_type().is_int()
    }

    pub fn is_float(&self) -> bool {
        self.as_type().is_float()
    }

    pub fn log2_lane_count(&self) -> u32 {
        self.as_type().log2_lane_count()
    }

    pub fn lane_count(&self) -> u32 {
        self.as_type().lane_count().into()
    }

    pub fn bits(&self) -> u32 {
        self.as_type().bits().into()
    }

    pub fn min_lane_count(&self) -> u32 {
        self.as_type().min_lane_count().into()
    }

    pub fn min_bits(&self) -> u32 {
        self.as_type().min_bits().into()
    }

    pub fn bytes(&self) -> u32 {
        self.as_type().bytes().into()
    }

    pub fn by(&self, n: u32) -> Option<Self> {
        self.as_type().by(n).map(Into::into)
    }

    pub fn vector_to_dynamic(&self) -> Option<Self> {
        self.as_type().vector_to_dynamic().map(Into::into)
    }

    pub fn dynamic_to_vector(&self) -> Option<Self> {
        self.as_type().dynamic_to_vector().map(Into::into)
    }

    pub fn split_lanes(&self) -> Option<Self> {
        self.as_type().split_lanes().map(Into::into)
    }

    pub fn merge_lanes(&self) -> Option<Self> {
        self.as_type().merge_lanes().map(Into::into)
    }

    pub fn index(&self) -> usize {
        self.as_type().index().into()
    }

    pub fn wider_or_equal(&self, other: Self) -> bool {
        self.as_type().wider_or_equal(other.as_type()).into()
    }

    #[staticmethod]
    pub fn target_pointer_type(target: &TargetIsa) -> Self {
        cl::codegen::ir::types::Type::triple_pointer_type(target.isa.triple()).into()
    }
}

impl Type {
    fn as_type(&self) -> cl::codegen::ir::types::Type {
        cl::codegen::ir::types::Type::from(*self)
    }
}

#[pyclass(eq, hash, frozen)]
#[derive(PartialEq, Debug, Clone, Copy, Hash)]
pub struct TrapCode(pub(crate) u8);

#[pymethods]
impl TrapCode {
    #[new]
    pub fn new(value: u8) -> PyResult<Self> {
        cl::codegen::ir::TrapCode::user(value)
            .ok_or(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid user trap code: {}",
                value
            )))
            .map(Into::into)
    }

    pub fn __repr__(&self) -> String {
        format!("TrapCode({:?})", self.0)
    }

    pub fn __str__(&self) -> String {
        format!("{}", cl::codegen::ir::TrapCode::from(*self))
    }
}

impl TrapCode {
    pub(crate) fn init_class(cls: Bound<'_, PyAny>) -> PyResult<()> {
        cls.setattr(
            "STACK_OVERFLOW",
            TrapCode::from(cl::codegen::ir::TrapCode::STACK_OVERFLOW),
        )?;
        cls.setattr(
            "HEAP_OUT_OF_BOUNDS",
            TrapCode::from(cl::codegen::ir::TrapCode::HEAP_OUT_OF_BOUNDS),
        )?;
        cls.setattr(
            "BAD_CONVERSION_TO_INTEGER",
            TrapCode::from(cl::codegen::ir::TrapCode::BAD_CONVERSION_TO_INTEGER),
        )?;
        cls.setattr(
            "INTEGER_DIVISION_BY_ZERO",
            TrapCode::from(cl::codegen::ir::TrapCode::INTEGER_DIVISION_BY_ZERO),
        )?;
        cls.setattr(
            "INTEGER_OVERFLOW",
            TrapCode::from(cl::codegen::ir::TrapCode::INTEGER_OVERFLOW),
        )?;
        Ok(())
    }
}

impl From<TrapCode> for cl::codegen::ir::TrapCode {
    fn from(value: TrapCode) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<cl::codegen::ir::TrapCode> for TrapCode {
    fn from(value: cl::codegen::ir::TrapCode) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

enum_wrap!(IntCC, cl::codegen::ir::condcodes::IntCC, {
    /// `==`.
    Equal,
    /// `!=`.
    NotEqual,
    /// Signed `<`.
    SignedLessThan,
    /// Signed `>=`.
    SignedGreaterThanOrEqual,
    /// Signed `>`.
    SignedGreaterThan,
    /// Signed `<=`.
    SignedLessThanOrEqual,
    /// Unsigned `<`.
    UnsignedLessThan,
    /// Unsigned `>=`.
    UnsignedGreaterThanOrEqual,
    /// Unsigned `>`.
    UnsignedGreaterThan,
    /// Unsigned `<=`.
    UnsignedLessThanOrEqual,
} {
    pub fn without_equal(&self) -> Self {
        self.as_intcc().without_equal().into()
    }

    pub fn unsigned(&self) -> Self {
        self.as_intcc().unsigned().into()
    }

    pub fn complement(&self) -> Self {
        self.as_intcc().complement().into()
    }

    pub fn swap_args(&self) -> Self {
        self.as_intcc().swap_args().into()
    }
});

impl IntCC {
    fn as_intcc(&self) -> cl::codegen::ir::condcodes::IntCC {
        cl::codegen::ir::condcodes::IntCC::from(*self)
    }
}

enum_wrap!(FloatCC, cl::codegen::ir::condcodes::FloatCC, {
    /// EQ | LT | GT
    Ordered,
    /// UN
    Unordered,

    /// EQ
    Equal,
    /// The C '!=' operator is the inverse of '==': `NotEqual`.
    /// UN | LT | GT
    NotEqual,
    /// LT | GT
    OrderedNotEqual,
    /// UN | EQ
    UnorderedOrEqual,

    /// LT
    LessThan,
    /// LT | EQ
    LessThanOrEqual,
    /// GT
    GreaterThan,
    /// GT | EQ
    GreaterThanOrEqual,

    /// UN | LT
    UnorderedOrLessThan,
    /// UN | LT | EQ
    UnorderedOrLessThanOrEqual,
    /// UN | GT
    UnorderedOrGreaterThan,
    /// UN | GT | EQ
    UnorderedOrGreaterThanOrEqual,
} {
    pub fn complement(&self) -> Self {
        self.as_floatcc().complement().into()
    }

    pub fn swap_args(&self) -> Self {
        self.as_floatcc().swap_args().into()
    }
});

impl FloatCC {
    fn as_floatcc(&self) -> cl::codegen::ir::condcodes::FloatCC {
        cl::codegen::ir::condcodes::FloatCC::from(*self)
    }
}

enum_wrap!(Endianness, cl::codegen::ir::Endianness, {
    /// Little-endian.
    Little,
    /// Big-endian.
    Big,
});

enum_wrap!(AliasRegion, cl::codegen::ir::AliasRegion, {
    Heap,
    Table,
    Vmctx,
});

enum_wrap!(AtomicRmwOp, cl::codegen::ir::AtomicRmwOp, {
    /// Add
    Add,
    /// Sub
    Sub,
    /// And
    And,
    /// Nand
    Nand,
    /// Or
    Or,
    /// Xor
    Xor,
    /// Exchange
    Xchg,
    /// Unsigned min
    Umin,
    /// Unsigned max
    Umax,
    /// Signed min
    Smin,
    /// Signed max
    Smax,
});
