use std::str::FromStr as _;

use crate::entities::*;
use cl::{
    codegen::{
        ir::{BlockCall, DynamicStackSlotData, KnownSymbol, LibCall},
        Context,
    },
    prelude::{
        AbiParam, ExtFuncData, ExternalName, InstBuilder as _, JumpTableData, StackSlotData,
    },
};
use ir::{MemFlags, Signature};
use pyo3::{pyclass, pymethods, types::PyAnyMethods as _, Bound, PyAny, PyResult};

pub mod ir;
pub mod isa;

#[pyclass]
pub struct FunctionBuilder {
    context: *mut cl::codegen::Context,
    fbc: *mut cl::frontend::FunctionBuilderContext,
    fb: cl::frontend::FunctionBuilder<'static>,
}

unsafe impl Send for FunctionBuilder {}
unsafe impl Sync for FunctionBuilder {}

#[pymethods]
impl FunctionBuilder {
    #[new]
    pub fn new() -> Self {
        let context: *mut cl::codegen::Context =
            Box::leak(Box::new(cl::codegen::Context::new())) as *mut _;
        let fbc = Box::leak(Box::new(cl::frontend::FunctionBuilderContext::new())) as *mut _;

        let func = unsafe { &mut (*context).func };
        let fb = cl::frontend::FunctionBuilder::new(func, unsafe { &mut *fbc });

        FunctionBuilder { context, fbc, fb }
    }

    pub fn signature_add_param(&mut self, ty: Type) {
        self.fb
            .func
            .stencil
            .signature
            .params
            .push(AbiParam::new(ty.into()));
    }

    pub fn signature_add_return(&mut self, ty: Type) {
        self.fb
            .func
            .stencil
            .signature
            .returns
            .push(AbiParam::new(ty.into()));
    }

    #[getter]
    pub fn get_func_signature(&self) -> Signature {
        self.fb.func.stencil.signature.clone().into()
    }

    #[setter]
    pub fn set_func_signature(&mut self, signature: &Signature) {
        self.fb.func.stencil.signature = signature.clone().into();
    }

    pub fn current_block(&self) -> Option<Block> {
        self.fb.current_block().map(|b| b.into())
    }

    // pub fn set_srcloc(&mut self, srcloc: SourceLoc) {
    //     self.fb.set_srcloc(srcloc.into());
    // }

    pub fn create_block(&mut self) -> Block {
        self.fb.create_block().into()
    }

    pub fn set_cold_block(&mut self, block: Block) {
        self.fb.set_cold_block(block.into());
    }

    pub fn insert_block_after(&mut self, block: Block, after: Block) {
        self.fb.insert_block_after(block.into(), after.into());
    }

    pub fn switch_to_block(&mut self, block: Block) {
        self.fb.switch_to_block(block.into());
    }

    pub fn seal_block(&mut self, block: Block) {
        self.fb.seal_block(block.into());
    }

    pub fn seal_all_blocks(&mut self) {
        self.fb.seal_all_blocks();
    }

    pub fn try_declare_var(&mut self, var: Variable, ty: Type) -> PyResult<()> {
        self.fb.try_declare_var(var.into(), ty.into()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to declare variable: {}", e))
        })
    }

    pub fn declare_var(&mut self, var: Variable, ty: Type) {
        self.fb.declare_var(var.into(), ty.into());
    }

    pub fn declare_var_needs_stack_map(&mut self, var: Variable) {
        self.fb.declare_var_needs_stack_map(var.into());
    }

    pub fn try_use_var(&mut self, var: Variable) -> PyResult<Value> {
        self.fb
            .try_use_var(var.into())
            .map(|v| v.into())
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to use variable: {}", e))
            })
    }

    pub fn use_var(&mut self, var: Variable) -> Value {
        self.fb.use_var(var.into()).into()
    }

    pub fn try_def_var(&mut self, var: Variable, val: Value) -> PyResult<()> {
        self.fb.try_def_var(var.into(), val.into()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to define variable: {}", e))
        })
    }

    pub fn def_var(&mut self, var: Variable, val: Value) {
        self.fb.def_var(var.into(), val.into());
    }

    pub fn set_val_label(&mut self, val: Value, label: ValueLabel) {
        self.fb.set_val_label(val.into(), label.into());
    }

    pub fn declare_value_needs_stack_map(&mut self, val: Value) {
        self.fb.declare_value_needs_stack_map(val.into());
    }

    pub fn create_jump_table(
        &mut self,
        default: (Block, Vec<Value>),
        data: Vec<(Block, Vec<Value>)>,
    ) -> JumpTable {
        fn to_block_call(
            fb: &mut cl::frontend::FunctionBuilder<'_>,
            v: (Block, Vec<Value>),
        ) -> BlockCall {
            BlockCall::new(
                v.0.into(),
                v.1.into_iter().map(Into::into),
                &mut fb.func.dfg.value_lists,
            )
        }

        let default = to_block_call(&mut self.fb, default);
        let data = data
            .into_iter()
            .map(|v| to_block_call(&mut self.fb, v))
            .collect::<Vec<_>>();

        self.fb
            .create_jump_table(JumpTableData::new(default, &data[..]))
            .into()
    }

    pub fn create_sized_stack_slot(&mut self, stack_size: u32, align_shift: u8) -> StackSlot {
        self.fb
            .create_sized_stack_slot(StackSlotData::new(
                cl::prelude::StackSlotKind::ExplicitSlot,
                stack_size,
                align_shift,
            ))
            .into()
    }

    pub fn create_dynamic_stack_slot(&mut self, stack_type: DynamicType) -> DynamicStackSlot {
        self.fb
            .create_dynamic_stack_slot(DynamicStackSlotData::new(
                cl::prelude::StackSlotKind::ExplicitDynamicSlot,
                stack_type.into(),
            ))
            .into()
    }

    pub fn import_signature(&mut self, signature: &Signature) -> SigRef {
        self.fb.import_signature(signature.as_ref().clone()).into()
    }

    #[pyo3(signature = (name, signature, colocated = false))]
    pub fn import_function(
        &mut self,
        name: &Bound<'_, PyAny>,
        signature: &SigRef,
        colocated: bool,
    ) -> PyResult<FuncRef> {
        let name = if let Ok(s) = name.extract::<&str>() {
            if let Ok(l) = LibCall::from_str(&s) {
                ExternalName::LibCall(l)
            } else if let Ok(k) = KnownSymbol::from_str(&s) {
                ExternalName::KnownSymbol(k)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected a string that is a valid LibCall or KnownSymbol",
                ));
            }
        } else if let Ok(un) = name.extract::<UserExternalNameRef>() {
            ExternalName::User(un.into())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected a string or UserExternalNameRef for function name",
            ));
        };

        Ok(self
            .fb
            .import_function(ExtFuncData {
                name: name.into(),
                signature: signature.clone().into(),
                colocated,
            })
            .into())
    }

    // pub fn create_global_value(&mut self, data: GlobalValueData) -> GlobalValue {
    //     self.fb.create_global_value(data.into()).into()
    // }

    pub fn ensure_inserted_block(&mut self) {
        self.fb.ensure_inserted_block();
    }

    // pub fn cursor(&mut self) -> FuncCursor {}

    pub fn append_block_params_for_function_params(&mut self, block: Block) {
        self.fb
            .append_block_params_for_function_params(block.into());
    }

    pub fn append_block_params_for_function_returns(&mut self, block: Block) {
        self.fb
            .append_block_params_for_function_returns(block.into());
    }

    pub fn finalize(&mut self) {
        let fb: cl::frontend::FunctionBuilder<'_> = unsafe { std::mem::transmute_copy(&self.fb) };
        fb.finalize();
        self.fb =
            cl::frontend::FunctionBuilder::new(unsafe { &mut (*self.context).func }, unsafe {
                &mut *self.fbc
            });
    }

    pub fn block_params(&self, block: Block) -> Vec<Value> {
        self.fb
            .block_params(block.into())
            .iter()
            .map(|v| (*v).into())
            .collect()
    }

    pub fn signature(&self, sigref: SigRef) -> Option<Signature> {
        self.fb
            .signature(sigref.into())
            .map(|sig| sig.clone().into())
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.fb.append_block_param(block.into(), ty.into()).into()
    }

    pub fn inst_results(&self, inst: Inst) -> Vec<Value> {
        self.fb
            .inst_results(inst.into())
            .iter()
            .map(|v| (*v).into())
            .collect()
    }

    pub fn change_jump_destination(&mut self, inst: Inst, old_block: Block, new_block: Block) {
        self.fb
            .change_jump_destination(inst.into(), old_block.into(), new_block.into());
    }

    pub fn is_unreachable(&self) -> bool {
        self.fb.is_unreachable()
    }

    // pub fn call_memcpy(}
    // pub fn emit_small_memory_copy(}
    // pub fn call_memset(}
    // pub fn emit_small_memset(}
    // pub fn call_memmove(}
    // pub fn call_memcmp(}
    // pub fn emit_small_memory_compare(}

    // ========== instructions ==========

    pub fn ins_jump(&mut self, block_call_label: Block, block_call_args: Vec<Value>) -> Inst {
        self.fb
            .ins()
            .jump(block_call_label.into(), &valvec(block_call_args))
            .into()
    }

    pub fn ins_brif(
        &mut self,
        c: Value,
        block_then_label: Block,
        block_then_args: Vec<Value>,
        block_else_label: Block,
        block_else_args: Vec<Value>,
    ) -> Inst {
        self.fb
            .ins()
            .brif(
                c.into(),
                block_then_label.into(),
                &valvec(block_then_args),
                block_else_label.into(),
                &valvec(block_else_args),
            )
            .into()
    }

    pub fn ins_br_table(&mut self, x: Value, jt: JumpTable) -> Inst {
        self.fb.ins().br_table(x.into(), jt.into()).into()
    }

    pub fn ins_debugtrap(&mut self) -> Inst {
        self.fb.ins().debugtrap().into()
    }

    pub fn ins_trap(&mut self, trap_code: TrapCode) -> Inst {
        self.fb.ins().trap(trap_code).into()
    }

    pub fn ins_trapz(&mut self, c: Value, trap_code: TrapCode) -> Inst {
        self.fb.ins().trapz(c.into(), trap_code).into()
    }

    pub fn ins_trapnz(&mut self, c: Value, trap_code: TrapCode) -> Inst {
        self.fb.ins().trapnz(c.into(), trap_code).into()
    }

    pub fn ins_return(&mut self, rvals: Vec<Value>) -> Inst {
        self.fb.ins().return_(&valvec(rvals)).into()
    }

    pub fn ins_call(&mut self, fn_ref: FuncRef, args: Vec<Value>) -> Inst {
        self.fb.ins().call(fn_ref.into(), &valvec(args)).into()
    }

    pub fn ins_call_indirect(&mut self, sig: SigRef, callee: Value, args: Vec<Value>) -> Inst {
        self.fb
            .ins()
            .call_indirect(sig.into(), callee.into(), &valvec(args))
            .into()
    }

    pub fn ins_return_call(&mut self, fn_ref: FuncRef, args: Vec<Value>) -> Inst {
        self.fb
            .ins()
            .return_call(fn_ref.into(), &valvec(args))
            .into()
    }

    pub fn ins_return_call_indirect(
        &mut self,
        sig: SigRef,
        callee: Value,
        args: Vec<Value>,
    ) -> Inst {
        self.fb
            .ins()
            .return_call_indirect(sig.into(), callee.into(), &valvec(args))
            .into()
    }

    pub fn ins_func_addr(&mut self, i_addr: Type, fn_ref: FuncRef) -> Value {
        self.fb.ins().func_addr(i_addr.into(), fn_ref.into()).into()
    }

    pub fn ins_splat(&mut self, txn: Type, x: Value) -> Value {
        self.fb.ins().splat(txn.into(), x.into()).into()
    }

    pub fn ins_swizzle(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().swizzle(x.into(), y.into()).into()
    }

    pub fn ins_x86_pshufb(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().x86_pshufb(x.into(), y.into()).into()
    }

    pub fn ins_insertlane(&mut self, x: Value, y: Value, imm: u8) -> Value {
        self.fb.ins().insertlane(x.into(), y.into(), imm).into()
    }

    pub fn ins_extractlane(&mut self, x: Value, imm: u8) -> Value {
        self.fb.ins().extractlane(x.into(), imm).into()
    }

    pub fn ins_smin(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().smin(x.into(), y.into()).into()
    }

    pub fn ins_umin(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().umin(x.into(), y.into()).into()
    }

    pub fn ins_smax(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().smax(x.into(), y.into()).into()
    }

    pub fn ins_umax(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().umax(x.into(), y.into()).into()
    }

    pub fn ins_avg_round(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().avg_round(x.into(), y.into()).into()
    }

    pub fn ins_uadd_sat(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().uadd_sat(x.into(), y.into()).into()
    }

    pub fn ins_sadd_sat(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().sadd_sat(x.into(), y.into()).into()
    }

    pub fn ins_usub_sat(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().usub_sat(x.into(), y.into()).into()
    }

    pub fn ins_ssub_sat(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().ssub_sat(x.into(), y.into()).into()
    }

    pub fn ins_load(&mut self, mem: Type, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb
            .ins()
            .load(mem.into(), mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_store(&mut self, mem_flags: MemFlags, x: Value, p: Value, offset32: i32) -> Inst {
        self.fb
            .ins()
            .store(mem_flags, x.into(), p.into(), offset32)
            .into()
    }

    pub fn ins_uload8(
        &mut self,
        i_ext8: Type,
        mem_flags: MemFlags,
        p: Value,
        offset32: i32,
    ) -> Value {
        self.fb
            .ins()
            .uload8(i_ext8.into(), mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_sload8(
        &mut self,
        i_ext8: Type,
        mem_flags: MemFlags,
        p: Value,
        offset32: i32,
    ) -> Value {
        self.fb
            .ins()
            .sload8(i_ext8.into(), mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_istore8(&mut self, mem_flags: MemFlags, x: Value, p: Value, offset32: i32) -> Inst {
        self.fb
            .ins()
            .istore8(mem_flags, x.into(), p.into(), offset32)
            .into()
    }

    pub fn ins_uload16(
        &mut self,
        i_ext16: Type,
        mem_flags: MemFlags,
        p: Value,
        offset32: i32,
    ) -> Value {
        self.fb
            .ins()
            .uload16(i_ext16.into(), mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_sload16(
        &mut self,
        i_ext16: Type,
        mem_flags: MemFlags,
        p: Value,
        offset32: i32,
    ) -> Value {
        self.fb
            .ins()
            .sload16(i_ext16.into(), mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_istore16(&mut self, mem_flags: MemFlags, x: Value, p: Value, offset32: i32) -> Inst {
        self.fb
            .ins()
            .istore16(mem_flags, x.into(), p.into(), offset32)
            .into()
    }

    pub fn ins_uload32(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb.ins().uload32(mem_flags, p.into(), offset32).into()
    }

    pub fn ins_sload32(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb.ins().sload32(mem_flags, p.into(), offset32).into()
    }

    pub fn ins_istore32(&mut self, mem_flags: MemFlags, x: Value, p: Value, offset32: i32) -> Inst {
        self.fb
            .ins()
            .istore32(mem_flags, x.into(), p.into(), offset32)
            .into()
    }

    pub fn ins_stack_switch(
        &mut self,
        store_context_ptr: Value,
        load_context_ptr: Value,
        in_payload0: Value,
    ) -> Value {
        self.fb
            .ins()
            .stack_switch(
                store_context_ptr.into(),
                load_context_ptr.into(),
                in_payload0.into(),
            )
            .into()
    }

    pub fn ins_uload8x8(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb.ins().uload8x8(mem_flags, p.into(), offset32).into()
    }

    pub fn ins_sload8x8(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb.ins().sload8x8(mem_flags, p.into(), offset32).into()
    }

    pub fn ins_uload16x4(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb
            .ins()
            .uload16x4(mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_sload16x4(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb
            .ins()
            .sload16x4(mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_uload32x2(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb
            .ins()
            .uload32x2(mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_sload32x2(&mut self, mem_flags: MemFlags, p: Value, offset32: i32) -> Value {
        self.fb
            .ins()
            .sload32x2(mem_flags, p.into(), offset32)
            .into()
    }

    pub fn ins_stack_load(&mut self, mem: Type, ss: StackSlot, offset32: i32) -> Value {
        self.fb
            .ins()
            .stack_load(mem.into(), ss.into(), offset32)
            .into()
    }

    pub fn ins_stack_store(&mut self, x: Value, ss: StackSlot, offset32: i32) -> Inst {
        self.fb
            .ins()
            .stack_store(x.into(), ss.into(), offset32)
            .into()
    }

    pub fn ins_stack_addr(&mut self, i_addr: Type, ss: StackSlot, offset32: i32) -> Value {
        self.fb
            .ins()
            .stack_addr(i_addr.into(), ss.into(), offset32)
            .into()
    }

    pub fn ins_dynamic_stack_load(&mut self, mem: Type, dss: DynamicStackSlot) -> Value {
        self.fb
            .ins()
            .dynamic_stack_load(mem.into(), dss.into())
            .into()
    }

    pub fn ins_dynamic_stack_store(&mut self, x: Value, dss: DynamicStackSlot) -> Inst {
        self.fb
            .ins()
            .dynamic_stack_store(x.into(), dss.into())
            .into()
    }

    pub fn ins_dynamic_stack_addr(&mut self, i_addr: Type, dss: DynamicStackSlot) -> Value {
        self.fb
            .ins()
            .dynamic_stack_addr(i_addr.into(), dss.into())
            .into()
    }

    pub fn ins_global_value(&mut self, mem: Type, gv: GlobalValue) -> Value {
        self.fb.ins().global_value(mem.into(), gv.into()).into()
    }

    pub fn ins_symbol_value(&mut self, mem: Type, gv: GlobalValue) -> Value {
        self.fb.ins().symbol_value(mem.into(), gv.into()).into()
    }

    pub fn ins_tls_value(&mut self, mem: Type, gv: GlobalValue) -> Value {
        self.fb.ins().tls_value(mem.into(), gv.into()).into()
    }

    pub fn ins_get_pinned_reg(&mut self, i_addr: Type) -> Value {
        self.fb.ins().get_pinned_reg(i_addr.into()).into()
    }

    pub fn ins_set_pinned_reg(&mut self, addr: Value) -> Inst {
        self.fb.ins().set_pinned_reg(addr.into()).into()
    }

    pub fn ins_get_frame_pointer(&mut self, i_addr: Type) -> Value {
        self.fb.ins().get_frame_pointer(i_addr.into()).into()
    }

    pub fn ins_get_stack_pointer(&mut self, i_addr: Type) -> Value {
        self.fb.ins().get_stack_pointer(i_addr.into()).into()
    }

    pub fn ins_get_return_address(&mut self, i_addr: Type) -> Value {
        self.fb.ins().get_return_address(i_addr.into()).into()
    }

    pub fn ins_iconst(&mut self, narrow_int: Type, imm: i64) -> Value {
        self.fb.ins().iconst(narrow_int.into(), imm).into()
    }
    // pub fn &mut self, ins_f16const(immf: f16) -> Value { self.fb.ins(). f16const().into() }

    pub fn ins_f32const(&mut self, imm: f32) -> Value {
        self.fb.ins().f32const(imm).into()
    }

    pub fn ins_f64const(&mut self, imm: f64) -> Value {
        self.fb.ins().f64const(imm).into()
    }
    // pub fn &mut self, ins_f128const<T1: Into<ir::Constant>>(N: T1) -> Value { self.fb.ins(). f128const().into() }

    pub fn ins_vconst(&mut self, txn: Type, constant: Constant) -> Value {
        self.fb.ins().vconst(txn.into(), constant).into()
    }

    pub fn ins_shuffle(&mut self, a: Value, b: Value, imm: Immediate) -> Value {
        self.fb.ins().shuffle(a.into(), b.into(), imm).into()
    }

    pub fn ins_nop(&mut self) -> Inst {
        self.fb.ins().nop().into()
    }

    pub fn ins_select(&mut self, c: Value, x: Value, y: Value) -> Value {
        self.fb.ins().select(c.into(), x.into(), y.into()).into()
    }

    pub fn ins_select_spectre_guard(&mut self, c: Value, x: Value, y: Value) -> Value {
        self.fb
            .ins()
            .select_spectre_guard(c.into(), x.into(), y.into())
            .into()
    }

    pub fn ins_bitselect(&mut self, c: Value, x: Value, y: Value) -> Value {
        self.fb.ins().bitselect(c.into(), x.into(), y.into()).into()
    }

    pub fn ins_x86_blendv(&mut self, c: Value, x: Value, y: Value) -> Value {
        self.fb
            .ins()
            .x86_blendv(c.into(), x.into(), y.into())
            .into()
    }

    pub fn ins_vany_true(&mut self, a: Value) -> Value {
        self.fb.ins().vany_true(a.into()).into()
    }

    pub fn ins_vall_true(&mut self, a: Value) -> Value {
        self.fb.ins().vall_true(a.into()).into()
    }

    pub fn ins_vhigh_bits(&mut self, narrow_int: Type, a: Value) -> Value {
        self.fb.ins().vhigh_bits(narrow_int.into(), a.into()).into()
    }

    pub fn ins_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Value {
        self.fb.ins().icmp(cc, x.into(), y.into()).into()
    }

    pub fn ins_icmp_imm(&mut self, cc: IntCC, x: Value, imm: i64) -> Value {
        self.fb.ins().icmp_imm(cc, x.into(), imm).into()
    }

    pub fn ins_iadd(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().iadd(x.into(), y.into()).into()
    }

    pub fn ins_isub(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().isub(x.into(), y.into()).into()
    }

    pub fn ins_ineg(&mut self, x: Value) -> Value {
        self.fb.ins().ineg(x.into()).into()
    }

    pub fn ins_iabs(&mut self, x: Value) -> Value {
        self.fb.ins().iabs(x.into()).into()
    }

    pub fn ins_imul(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().imul(x.into(), y.into()).into()
    }

    pub fn ins_umulhi(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().umulhi(x.into(), y.into()).into()
    }

    pub fn ins_smulhi(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().smulhi(x.into(), y.into()).into()
    }

    pub fn ins_sqmul_round_sat(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().sqmul_round_sat(x.into(), y.into()).into()
    }

    pub fn ins_x86_pmulhrsw(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().x86_pmulhrsw(x.into(), y.into()).into()
    }

    pub fn ins_udiv(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().udiv(x.into(), y.into()).into()
    }

    pub fn ins_sdiv(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().sdiv(x.into(), y.into()).into()
    }

    pub fn ins_urem(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().urem(x.into(), y.into()).into()
    }

    pub fn ins_srem(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().srem(x.into(), y.into()).into()
    }

    pub fn ins_iadd_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().iadd_imm(x.into(), imm).into()
    }

    pub fn ins_imul_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().imul_imm(x.into(), imm).into()
    }

    pub fn ins_udiv_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().udiv_imm(x.into(), imm).into()
    }

    pub fn ins_sdiv_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().sdiv_imm(x.into(), imm).into()
    }

    pub fn ins_urem_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().urem_imm(x.into(), imm).into()
    }

    pub fn ins_srem_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().srem_imm(x.into(), imm).into()
    }

    pub fn ins_irsub_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().irsub_imm(x.into(), imm).into()
    }

    pub fn ins_sadd_overflow_cin(&mut self, x: Value, y: Value, c_in: Value) -> (Value, Value) {
        let (x, y) = self
            .fb
            .ins()
            .sadd_overflow_cin(x.into(), y.into(), c_in.into());
        (x.into(), y.into())
    }

    pub fn ins_uadd_overflow_cin(&mut self, x: Value, y: Value, c_in: Value) -> (Value, Value) {
        let (x, y) = self
            .fb
            .ins()
            .uadd_overflow_cin(x.into(), y.into(), c_in.into());
        (x.into(), y.into())
    }

    pub fn ins_uadd_overflow(&mut self, x: Value, y: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().uadd_overflow(x.into(), y.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_sadd_overflow(&mut self, x: Value, y: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().sadd_overflow(x.into(), y.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_usub_overflow(&mut self, x: Value, y: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().usub_overflow(x.into(), y.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_ssub_overflow(&mut self, x: Value, y: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().ssub_overflow(x.into(), y.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_umul_overflow(&mut self, x: Value, y: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().umul_overflow(x.into(), y.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_smul_overflow(&mut self, x: Value, y: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().smul_overflow(x.into(), y.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_uadd_overflow_trap(&mut self, x: Value, y: Value, trap_code: TrapCode) -> Value {
        self.fb
            .ins()
            .uadd_overflow_trap(x.into(), y.into(), trap_code)
            .into()
    }

    pub fn ins_ssub_overflow_bin(&mut self, x: Value, y: Value, b_in: Value) -> (Value, Value) {
        let (x, y) = self
            .fb
            .ins()
            .ssub_overflow_bin(x.into(), y.into(), b_in.into());
        (x.into(), y.into())
    }

    pub fn ins_usub_overflow_bin(&mut self, x: Value, y: Value, b_in: Value) -> (Value, Value) {
        let (x, y) = self
            .fb
            .ins()
            .usub_overflow_bin(x.into(), y.into(), b_in.into());
        (x.into(), y.into())
    }

    pub fn ins_band(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().band(x.into(), y.into()).into()
    }

    pub fn ins_bor(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().bor(x.into(), y.into()).into()
    }

    pub fn ins_bxor(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().bxor(x.into(), y.into()).into()
    }

    pub fn ins_bnot(&mut self, x: Value) -> Value {
        self.fb.ins().bnot(x.into()).into()
    }

    pub fn ins_band_not(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().band_not(x.into(), y.into()).into()
    }

    pub fn ins_bor_not(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().bor_not(x.into(), y.into()).into()
    }

    pub fn ins_bxor_not(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().bxor_not(x.into(), y.into()).into()
    }

    pub fn ins_band_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().band_imm(x.into(), imm).into()
    }

    pub fn ins_bor_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().bor_imm(x.into(), imm).into()
    }

    pub fn ins_bxor_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().bxor_imm(x.into(), imm).into()
    }

    pub fn ins_rotl(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().rotl(x.into(), y.into()).into()
    }

    pub fn ins_rotr(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().rotr(x.into(), y.into()).into()
    }

    pub fn ins_rotl_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().rotl_imm(x.into(), imm).into()
    }

    pub fn ins_rotr_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().rotr_imm(x.into(), imm).into()
    }

    pub fn ins_ishl(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().ishl(x.into(), y.into()).into()
    }

    pub fn ins_ushr(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().ushr(x.into(), y.into()).into()
    }

    pub fn ins_sshr(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().sshr(x.into(), y.into()).into()
    }

    pub fn ins_ishl_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().ishl_imm(x.into(), imm).into()
    }

    pub fn ins_ushr_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().ushr_imm(x.into(), imm).into()
    }

    pub fn ins_sshr_imm(&mut self, x: Value, imm: i64) -> Value {
        self.fb.ins().sshr_imm(x.into(), imm).into()
    }

    pub fn ins_bitrev(&mut self, x: Value) -> Value {
        self.fb.ins().bitrev(x.into()).into()
    }

    pub fn ins_clz(&mut self, x: Value) -> Value {
        self.fb.ins().clz(x.into()).into()
    }

    pub fn ins_cls(&mut self, x: Value) -> Value {
        self.fb.ins().cls(x.into()).into()
    }

    pub fn ins_ctz(&mut self, x: Value) -> Value {
        self.fb.ins().ctz(x.into()).into()
    }

    pub fn ins_bswap(&mut self, x: Value) -> Value {
        self.fb.ins().bswap(x.into()).into()
    }

    pub fn ins_popcnt(&mut self, x: Value) -> Value {
        self.fb.ins().popcnt(x.into()).into()
    }

    pub fn ins_fcmp(&mut self, cc: FloatCC, x: Value, y: Value) -> Value {
        self.fb.ins().fcmp(cc, x.into(), y.into()).into()
    }

    pub fn ins_fadd(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fadd(x.into(), y.into()).into()
    }

    pub fn ins_fsub(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fsub(x.into(), y.into()).into()
    }

    pub fn ins_fmul(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fmul(x.into(), y.into()).into()
    }

    pub fn ins_fdiv(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fdiv(x.into(), y.into()).into()
    }

    pub fn ins_sqrt(&mut self, x: Value) -> Value {
        self.fb.ins().sqrt(x.into()).into()
    }

    pub fn ins_fma(&mut self, x: Value, y: Value, z: Value) -> Value {
        self.fb.ins().fma(x.into(), y.into(), z.into()).into()
    }

    pub fn ins_fneg(&mut self, x: Value) -> Value {
        self.fb.ins().fneg(x.into()).into()
    }

    pub fn ins_fabs(&mut self, x: Value) -> Value {
        self.fb.ins().fabs(x.into()).into()
    }

    pub fn ins_fcopysign(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fcopysign(x.into(), y.into()).into()
    }

    pub fn ins_fmin(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fmin(x.into(), y.into()).into()
    }

    pub fn ins_fmax(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().fmax(x.into(), y.into()).into()
    }

    pub fn ins_ceil(&mut self, x: Value) -> Value {
        self.fb.ins().ceil(x.into()).into()
    }

    pub fn ins_floor(&mut self, x: Value) -> Value {
        self.fb.ins().floor(x.into()).into()
    }

    pub fn ins_trunc(&mut self, x: Value) -> Value {
        self.fb.ins().trunc(x.into()).into()
    }

    pub fn ins_nearest(&mut self, x: Value) -> Value {
        self.fb.ins().nearest(x.into()).into()
    }

    pub fn ins_bitcast(&mut self, mem_to: Type, mem_flags: MemFlags, x: Value) -> Value {
        self.fb
            .ins()
            .bitcast(mem_to.into(), mem_flags, x.into())
            .into()
    }

    pub fn ins_scalar_to_vector(&mut self, txn: Type, s: Value) -> Value {
        self.fb.ins().scalar_to_vector(txn.into(), s.into()).into()
    }

    pub fn ins_bmask(&mut self, int_to: Type, x: Value) -> Value {
        self.fb.ins().bmask(int_to.into(), x.into()).into()
    }

    pub fn ins_ireduce(&mut self, int: Type, x: Value) -> Value {
        self.fb.ins().ireduce(int.into(), x.into()).into()
    }

    pub fn ins_snarrow(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().snarrow(x.into(), y.into()).into()
    }

    pub fn ins_unarrow(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().unarrow(x.into(), y.into()).into()
    }

    pub fn ins_uunarrow(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().uunarrow(x.into(), y.into()).into()
    }

    pub fn ins_swiden_low(&mut self, x: Value) -> Value {
        self.fb.ins().swiden_low(x.into()).into()
    }

    pub fn ins_swiden_high(&mut self, x: Value) -> Value {
        self.fb.ins().swiden_high(x.into()).into()
    }

    pub fn ins_uwiden_low(&mut self, x: Value) -> Value {
        self.fb.ins().uwiden_low(x.into()).into()
    }

    pub fn ins_uwiden_high(&mut self, x: Value) -> Value {
        self.fb.ins().uwiden_high(x.into()).into()
    }

    pub fn ins_iadd_pairwise(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().iadd_pairwise(x.into(), y.into()).into()
    }

    pub fn ins_x86_pmaddubsw(&mut self, x: Value, y: Value) -> Value {
        self.fb.ins().x86_pmaddubsw(x.into(), y.into()).into()
    }

    pub fn ins_uextend(&mut self, int: Type, x: Value) -> Value {
        self.fb.ins().uextend(int.into(), x.into()).into()
    }

    pub fn ins_sextend(&mut self, int: Type, x: Value) -> Value {
        self.fb.ins().sextend(int.into(), x.into()).into()
    }

    pub fn ins_fpromote(&mut self, float_scalar: Type, x: Value) -> Value {
        self.fb.ins().fpromote(float_scalar.into(), x.into()).into()
    }

    pub fn ins_fdemote(&mut self, float_scalar: Type, x: Value) -> Value {
        self.fb.ins().fdemote(float_scalar.into(), x.into()).into()
    }

    pub fn ins_fvdemote(&mut self, x: Value) -> Value {
        self.fb.ins().fvdemote(x.into()).into()
    }

    pub fn ins_fvpromote_low(&mut self, a: Value) -> Value {
        self.fb.ins().fvpromote_low(a.into()).into()
    }

    pub fn ins_fcvt_to_uint(&mut self, int_to: Type, x: Value) -> Value {
        self.fb.ins().fcvt_to_uint(int_to.into(), x.into()).into()
    }

    pub fn ins_fcvt_to_sint(&mut self, int_to: Type, x: Value) -> Value {
        self.fb.ins().fcvt_to_sint(int_to.into(), x.into()).into()
    }

    pub fn ins_fcvt_to_uint_sat(&mut self, int_to: Type, x: Value) -> Value {
        self.fb
            .ins()
            .fcvt_to_uint_sat(int_to.into(), x.into())
            .into()
    }

    pub fn ins_fcvt_to_sint_sat(&mut self, int_to: Type, x: Value) -> Value {
        self.fb
            .ins()
            .fcvt_to_sint_sat(int_to.into(), x.into())
            .into()
    }

    pub fn ins_x86_cvtt2dq(&mut self, int_to: Type, x: Value) -> Value {
        self.fb.ins().x86_cvtt2dq(int_to.into(), x.into()).into()
    }

    pub fn ins_fcvt_from_uint(&mut self, float_to: Type, x: Value) -> Value {
        self.fb
            .ins()
            .fcvt_from_uint(float_to.into(), x.into())
            .into()
    }

    pub fn ins_fcvt_from_sint(&mut self, float_to: Type, x: Value) -> Value {
        self.fb
            .ins()
            .fcvt_from_sint(float_to.into(), x.into())
            .into()
    }

    pub fn ins_isplit(&mut self, x: Value) -> (Value, Value) {
        let (x, y) = self.fb.ins().isplit(x.into()).into();
        (x.into(), y.into())
    }

    pub fn ins_iconcat(&mut self, lo: Value, hi: Value) -> Value {
        self.fb.ins().iconcat(lo.into(), hi.into()).into()
    }

    pub fn ins_atomic_rmw(
        &mut self,
        ty: Type,
        mem_flags: MemFlags,
        op: AtomicRmwOp,
        p: Value,
        x: Value,
    ) -> Value {
        self.fb
            .ins()
            .atomic_rmw(ty.into(), mem_flags, op, p.into(), x.into())
            .into()
    }

    pub fn ins_atomic_cas(&mut self, mem_flags: MemFlags, p: Value, e: Value, x: Value) -> Value {
        self.fb
            .ins()
            .atomic_cas(mem_flags, p.into(), e.into(), x.into())
            .into()
    }

    pub fn ins_atomic_load(&mut self, ty: Type, mem_flags: MemFlags, p: Value) -> Value {
        self.fb
            .ins()
            .atomic_load(ty.into(), mem_flags, p.into())
            .into()
    }

    pub fn ins_atomic_store(&mut self, mem_flags: MemFlags, x: Value, p: Value) -> Inst {
        self.fb
            .ins()
            .atomic_store(mem_flags, x.into(), p.into())
            .into()
    }

    pub fn ins_fence(&mut self) -> Inst {
        self.fb.ins().fence().into()
    }

    pub fn ins_extract_vector(&mut self, x: Value, imm: u8) -> Value {
        self.fb.ins().extract_vector(x.into(), imm).into()
    }
}

fn valvec<T>(v: Vec<Value>) -> Vec<T>
where
    Value: Into<T>,
{
    v.into_iter().map(Into::into).collect()
}

impl AsMut<Context> for FunctionBuilder {
    fn as_mut(&mut self) -> &mut Context {
        unsafe { &mut (*self.context) }
    }
}

impl Drop for FunctionBuilder {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.fbc));
            drop(Box::from_raw(self.context));
        }
    }
}
