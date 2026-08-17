"""The TOG must describe the loops triton-npu actually emits.

Reports where the producer stops rather than asserting success, in the style of
test_triton_codegen.py. The fixture is a post-vcix kernel in the shape tnpu
emits today: an i32 `scf.for` holding a DMA and one systolic push/pop group.

    python tests/system/test_tog_structure.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: One reduction trip: an MVIN whose offset comes off the loop variable through
#: arith (not affine.apply), then weight preload / input push / pop.
FIXTURE = """
module {
  memref.global @buf0_spad_128lane : memref<128x128xf32, 1> {tnpu.lane_axis = 1 : i64}
  memref.global @buf1_spad_1lane : memref<1xi32, 1> {tnpu.lane_axis = 0 : i64}
  func.func @tog_fixture_kernel(%arg0: memref<*xf32>, %pid: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %vl = arith.constant 8 : i64
    %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
    %spad = memref.get_global @buf0_spad_128lane : memref<128x128xf32, 1>
    %tag = memref.get_global @buf1_spad_1lane : memref<1xi32, 1>
    %dram = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16384], strides: [1] : memref<*xf32> to memref<16384xf32>
    scf.for %k = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %ki = arith.index_cast %k : i32 to index
      %off = arith.muli %ki, %c128 : index
      "togsim.transfer"(%dram, %off, %spad, %c0, %tag, %c0, %c1) {dma_kind = "MVIN", dram_arg = 0 : i64, dram_stride = [128, 1]} : (memref<16384xf32>, index, memref<128x128xf32, 1>, index, memref<1xi32, 1>, index, index) -> ()
      %w = arith.bitcast %cst : vector<8xf32> to vector<8xi32>
      %op_pre = llvm.mlir.constant(1 : i64) : i64
      %rd = llvm.mlir.constant(0 : i64) : i64
      %imm = llvm.mlir.constant(0 : i64) : i64
      llvm.call_intrinsic "llvm.riscv.sf.vc.iv.se"(%op_pre, %rd, %w, %imm, %vl) : (i64, i64, vector<8xi32>, i64, i64) -> ()
      %op_mm = llvm.mlir.constant(0 : i64) : i64
      llvm.call_intrinsic "llvm.riscv.sf.vc.iv.se"(%op_mm, %rd, %w, %imm, %vl) : (i64, i64, vector<8xi32>, i64, i64) -> ()
      %op_pop = llvm.mlir.constant(2 : i64) : i64
      %pop = llvm.call_intrinsic "llvm.riscv.sf.vc.v.i.se"(%op_pop, %rd, %imm, %vl) : (i64, i64, i64, i64) -> vector<8xi64>
    }
    return
  }
}
"""

MATMUL_COMPUTE = 1
MATMUL_PRELOAD = 2


def _parse(src):
    from PyTorchSimFrontend.tog.build_tog import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx, ir.Module.parse(src, ctx)


def _togsim_ops(module, name):
    from PyTorchSimFrontend.tog._mlir_util import walk_ops

    return [op for op in walk_ops(module.body) if op.operation.name == name]


def check_loop_body_is_read():
    """A DMA inside a loop must reach the trace, and the pushes must be typed.

    Everything here is dropped today: the walker only descends into a loop
    carrying an MLIR-route role attribute, which triton-npu never writes.
    """
    from PyTorchSimFrontend.tog import build_skeleton as bs
    from PyTorchSimFrontend.tog import togsim_ops as ts
    from PyTorchSimFrontend.tog.build_tog import ir

    problems = []
    ctx, module = _parse(FIXTURE)
    with ctx:
        want_dma = len(_togsim_ops(module, "togsim.transfer"))
        try:
            bs.build_skeleton(module)
        except Exception as e:  # noqa: BLE001
            print(f"  loop body: build_skeleton raised {type(e).__name__}: {e}")
            return False

        got_dma = len(_togsim_ops(module, ts.DMA))
        if got_dma != want_dma:
            problems.append(f"{want_dma} togsim.transfer became {got_dma} togsim.dma")

        types = sorted(
            ir.IntegerAttr(op.operation.attributes[ts.ATTR_COMPUTE_TYPE]).value
            for op in _togsim_ops(module, ts.COMPUTE))
        for want, what in ((MATMUL_PRELOAD, "preload"), (MATMUL_COMPUTE, "matmul")):
            if want not in types:
                problems.append(f"no {what} compute node (types {types})")

    for p in problems:
        print(f"  loop body: {p}")
    return not problems


def check_sampled_loops_are_the_replayed_loops():
    """Every loop the trace replays must be one gem5 samples a single trip of.

    A loop that survives without a node is measured whole and replayed anyway
    (double count); a node whose loop is folded away is measured once and never
    replayed (undercount). Only equality is right.
    """
    from PyTorchSimFrontend.tog import build_skeleton as bs
    from PyTorchSimFrontend.tog import lower_to_emitc as l2e
    from PyTorchSimFrontend.tog.build_tog import (TogBuilder, _build, _reset_ids)
    from PyTorchSimFrontend.tog._mlir_util import walk_ops

    problems = []
    ctx, module = _parse(FIXTURE)
    with ctx:
        _reset_ids()
        builder = TogBuilder()
        try:
            _build(module, builder)
        except Exception as e:  # noqa: BLE001
            print(f"  loop nodes: _build raised {type(e).__name__}: {e}")
            return False
        nodes = len(builder.loop_nodes)

    ctx, module = _parse(FIXTURE)
    with ctx:
        try:
            bs.build_skeleton(module)
            survived = sum(1 for op in walk_ops(module.body)
                           if op.operation.name in ("affine.for", "scf.for"))
            emitc = l2e.lower_to_emitc(
                module, work_item=l2e.WorkItem(parallel_args=[1], grid=[4]))
            cpp = l2e.emitc_to_cpp(emitc, include_dir=l2e._default_include_dir())
        except Exception as e:  # noqa: BLE001
            print(f"  loop nodes: lowering raised {type(e).__name__}: {e}")
            return False

    if nodes != survived:
        problems.append(f"{nodes} loop node(s) but {survived} loop(s) in the skeleton")
    if cpp.count("for (") < 2:
        problems.append(f"expected the grid loop and the reduction loop, "
                        f"found {cpp.count('for (')} loop(s) in the C++")

    for p in problems:
        print(f"  loop nodes: {p}")
    return not problems


if __name__ == "__main__":
    checks = [("loop body is read", check_loop_body_is_read),
              ("sampled loops == replayed loops",
               check_sampled_loops_are_the_replayed_loops)]
    failed = []
    for name, fn in checks:
        print(f"[*] {name}")
        if not fn():
            failed.append(name)
    print()
    if failed:
        print("|TOG structure Test Failed| " + ", ".join(failed))
        sys.exit(1)
    print("|TOG structure Test Passed|")
