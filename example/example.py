from cranelift import *
from cffi import FFI


def main():

    sig = Signature(CallConv.SystemV)
    sig.add_return(Type.I32)

    fb = FunctionBuilder()
    fb.func_signature = sig

    b = fb.create_block()
    fb.switch_to_block(b)

    lhs = fb.ins_iconst(Type.I32, 1)
    rhs = fb.ins_iconst(Type.I32, 2)
    sum = fb.ins_iadd(lhs, rhs)
    fb.ins_return([sum])

    fb.seal_all_blocks()
    fb.finalize()

    module = JITModule()
    func_id = module.declare_function("add_numbers", Linkage.Export, sig)
    module.define_function(func_id, fb)

    module.finalize_definitions()
    f_ptr = module.get_finalized_function(func_id)

    ffi = FFI()
    r = ffi.cast("int (*)()", f_ptr)
    result = r()
    print("1 + 2 =", result)
    assert result == 3, f"Expected 3, got {result}"


if __name__ == "__main__":
    main()
