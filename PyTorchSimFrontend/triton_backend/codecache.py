"""Compile cache for the Triton route -- the counterpart of extension_codecache.

`triton_npu_compile` is what the generated wrapper calls, exactly where the MLIR
route calls `custom_async_compile.mlir(...)`. It compiles the Triton kernel via
tnpu and returns the callable the wrapper then invokes per launch.

    define_kernel   ->  triton_npu_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

Layout mirrors the MLIR route so the two are comparable: one directory per source
hash under the dump path, holding the generated tnpu kernel file and every tnpu
artifact (01-ttir.mlir ... *-<kernel>.elf).
"""

import os
import re

from filelock import FileLock
from torch._inductor.codecache import get_hash

from PyTorchSimFrontend import extension_config
from . import functional, kernel_spec, timing, tnpu_bridge

logger = extension_config.setup_logger()

LOCK_TIMEOUT = 600


def _write_path(src_code):
    return os.path.join(extension_config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip())[1:12])


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper.

    Holds the compile result; each call is one launch of the whole grid.
    """

    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta

    def __call__(self, *args):
        """One launch of the whole grid: run it on Spike, then time it.

        Spike runs first so the caller's output tensors hold real values even if
        TOGSim fails -- the two halves are independent.
        """
        if extension_config.pytorchsim_functional_mode:
            written = functional.run(self.workdir, self.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        # AND THE OTHER HALF IS SWITCHED TOO, which the paragraph above already
        # claims: "the two halves are independent". Only the functional one was
        # -- the timing half ran whatever the config said, so a graph being
        # checked for VALUES paid for a cycle simulation of every kernel it
        # touched. That is the whole cost of an e2e run: mobilenet_v2's
        # depthwise convolutions launch a grid of [144, 2, 49] each, and the
        # model took over two hours to reach kernel 16 of 57 with timing on and
        # minutes with it off. `pytorchsim_timing_mode` is the switch the MLIR
        # route already reads (extension_codecache.py), so this route reads the
        # same one rather than inventing a second name.
        # AND THE MEASUREMENT FROM THE OTHER DIRECTION. Before this switch
        # existed the timing half ran even when only correctness was wanted, so
        # a kernel whose TIMING path failed failed the whole launch though Spike
        # had written the right values -- which is the state
        # tests/ops/attention/test_gqa.py and
        # tests/ops/fusion/test_prologue_fusion.py were in: their bmm template
        # kernels ran on Spike and died in emit_trace.
        if not extension_config.pytorchsim_timing_mode:
            # NOT "[TOGSim]". The sweep buckets a failure by matching its
            # output, and its togsim bucket is `TOGSim|trace\.so|SIGSEGV|...` --
            # so a line carrying that word puts every failing test in this mode
            # into the wrong bucket whatever actually went wrong. MEASURED:
            # tests/system/test_triton_codegen.py came back "[togsim]" for a
            # failure that had nothing to do with it.
            logger.warning(
                "[timing] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        if not os.path.isfile(os.path.join(self.workdir, timing.TRACE_SO)):
            timing.emit_trace(self.workdir, self.meta)
        result = timing.run_togsim(self.workdir, meta=self.meta, args=args)
        logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


#: tnpu's machine-readable half of a scratchpad refusal (tnpu/spad.py,
#: SPAD_OVERFLOW_MARKER). The rest of that message is advice for a person.
_SPAD_OVERFLOW_RE = re.compile(
    r"tnpu-spad-overflow: usage=(\d+) budget=(\d+)")


def _spad_overflow(exc):
    """(usage, budget) if this failure was the scratchpad, else None.

    READ OFF `exc.output`, NOT `str(exc)`. TnpuError's message is a summary --
    it keeps the last few lines that look like a diagnostic (`error:`, an
    exception name, an assertion) so Inductor, which prints only `str(exc)`,
    shows something useful. The marker looks like none of those on purpose: it
    is addressed to this function, not to a reader, and widening that filter to
    let it through would put it in front of the reader instead. The raw stage
    output is where a contract belongs.
    """
    m = _SPAD_OVERFLOW_RE.search(getattr(exc, "output", None) or str(exc))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _shrink_tile(meta, usage, budget):
    """Divide the tile by enough to fit, in place. False if stuck.

    WHY THE BLOCK SIZE IS THE FREE VARIABLE AND THE BUFFER COUNT IS NOT.
    `fixed_config_for` sizes R0_BLOCK against a budget divided by
    `_REDUCTION_LIVE_TILES`, a constant standing in for how many block-sized
    tiles the kernel will keep live -- and that count is a property of the
    LOWERING, which does not exist until the block size has been chosen. The
    constant's own comment says so ("the count is a property of the kernel and
    not a constant") and then guesses anyway, at 12. ViT's first LayerNorm,
    fused with a patch convolution, an addmm and a transpose, lowers to 41
    scratchpad globals:

        R0_BLOCK 512   77504 bytes/lane   over the 65536 budget
        R0_BLOCK 256   38680             fits
        R0_BLOCK 128   19356

    So the guess cannot be made right by picking a bigger number -- 41 would
    cost every ordinary reduction three quarters of its tile. It can only be
    CORRECTED, and the correction is one recompile: tnpu measures the real
    thing and says by how much.

    `usage / budget` rounded up to a power of two, so an overshoot of 1.18x
    halves once and an overshoot of 5x goes straight to an eighth rather than
    walking there. A reduction block moves first: XBLOCK is usually the lane
    axis, and shrinking it would leave lanes idle without freeing a byte.

    ONLY A BLOCK THE KERNEL ACCEPTS AS AN ARGUMENT MOVES ANYTHING. A PERSISTENT
    reduction -- Inductor's choice when r0_numel is small enough to hold the
    whole reduction in one tile -- writes the block size into the kernel BODY:

        @triton_heuristics.persistent_reduction(size_hints={'x': 1024, 'r0_': 128})
        def triton_npu_fused_..._5(..., XBLOCK : tl.constexpr):
            R0_BLOCK: tl.constexpr = 128        # <- not an argument

    so R0_BLOCK is absent from the signature and rewriting it here changes the
    spec, recompiles, and produces THE SAME KERNEL. Measured on Qwen3's fused
    q_norm + rope (`_unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_...`, 27
    scratchpad globals over a [XBLOCK, 128] tile), where the loop used to run
    down to R0_BLOCK 1 and fail with the overflow unchanged at every step:

        R0_BLOCK 128 -> 8 -> 1   1625120 bytes/lane, all three, over 131072
        XBLOCK   128 -> 64        fits (and 16 and 8 also compile)

    In that kernel the reduction axis is the vectorised one -- tnpu names the
    globals `bufN_spad_1lane` and Inductor scores the tiling `{'x': 0,
    'r0_': 2655744}` -- so X is the tile's outer axis and shrinking it is
    exactly what frees bytes per lane. Hence: reduction blocks if the kernel
    takes any, XBLOCK only when it takes none, and never both.
    """
    factor = 1
    while usage > budget * factor:
        factor *= 2
    cfg = meta.get("fixed_config") or {}
    signature = meta.get("signature") or {}

    def _movable(names):
        return {k: v for k, v in cfg.items()
                if k in names and k in signature and v and v > 1}

    blocks = _movable([k for k in cfg
                       if k.startswith("R") and k.endswith("_BLOCK")])
    if not blocks:
        blocks = _movable(["XBLOCK"])
    if not blocks:
        return False
    for k, v in blocks.items():
        cfg[k] = max(1, v // factor)
    return True


def triton_npu_compile(src_code, meta, kernel_name):
    """Compile one Inductor-generated Triton kernel through tnpu.

    Called from the generated wrapper at module import time (same point as
    `custom_async_compile.mlir`). Synchronous for now: the MLIR route's thread
    pool buys nothing until the pipeline itself is proven.
    """
    write_path = _write_path(src_code)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = tnpu_bridge.stage_artifact(write_path, f"{kernel_name}.elf")
        if elf is None:
            # Before write_spec_file, which rejects exactly the kernels whose
            # source is worth keeping.
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)      # the unmodified Inductor source
            timing.store_meta(write_path, meta)   # lets the timing step run standalone
            last_usage = None
            while True:
                kernel_spec.write_spec_file(src_code, meta, spec_path,
                                            tnpu_bridge.tnpu_dir())
                try:
                    tnpu_bridge.run_pipeline(spec_path, write_path,
                                             to_stage="binary")
                    break
                except tnpu_bridge.TnpuError as exc:
                    over = _spad_overflow(exc)
                    if over is None:
                        raise
                    # A RETRY THAT FREES NOTHING IS NOT A RETRY, and walking on
                    # anyway is worse than stopping: the tile keeps halving, the
                    # measurement does not move, and the last halving is the one
                    # that takes the tile's outer axis to 1. A unit axis is not
                    # an axis, so select_lane_axis has to put the lanes
                    # somewhere else and does not answer the same way for every
                    # buffer -- the fold then crosses lanes, which this machine
                    # has no instruction for, and the kernel COMPILES and
                    # returns wrong numbers.
                    #
                    #   measured  Qwen2-MoE's sort kernel, spad 131072, each
                    #             size compiled on its own and the ELF's .spad
                    #             section read back:
                    #
                    #               XBLOCK 128   273,936   axis 0 throughout
                    #                      32    149,264   axis 0
                    #                      16    132,592   axis 0
                    #                       8    127,616   axis 0
                    #                       4    127,616   axis 0
                    #                       2    127,616   axis 0
                    #                       1     37,792   0 x300, 1 x125, ONE_LANE x4
                    #
                    # AND BOTH HALVES OF THAT TABLE HAVE A REASON. 8, 4 and 2
                    # are the same number because `spad.reserve_per_lane` rounds
                    # every reservation up to MIN_VEC -- a lane holding 8, 4 or
                    # 2 elements has a whole 8-element vector written to it
                    # either way -- so below 8 there is nothing left for XBLOCK
                    # to free. And 1 does not fit because the tile got smaller:
                    # it fits because the BANKING COLLAPSED. 125 buffers moved
                    # onto the sorted axis, where a lane holds one element
                    # instead of a row, and the reservation fell with them. The
                    # model came out at Max abs diff 1.369991421699524 -- twice,
                    # to the last digit, at two scratchpad sizes -- against a
                    # 1e-6 target, while XBLOCK 8 and 64 both pass.
                    #
                    # So the only size that ever got under the budget got there
                    # by breaking the kernel. Stopping where the bytes stop
                    # moving is stopping exactly before that.
                    #
                    # So stop at the last size that changed something and let
                    # the overflow be reported. A compile error names the kernel
                    # and the budget; a wrong number names nothing.
                    if last_usage is not None and over[0] >= last_usage:
                        logger.warning(
                            "[triton-npu] %s: %d bytes/lane, unchanged from the "
                            "previous tile -- shrinking further frees nothing "
                            "and only risks a unit axis, so stopping here",
                            kernel_name, over[0])
                        raise
                    last_usage = over[0]
                    if not _shrink_tile(meta, *over):
                        raise
                    # EVERY BLOCK, and the filter used to be `endswith("_BLOCK")`
                    # -- which is false of "XBLOCK", the only block that moves in
                    # a persistent reduction (see _shrink_tile). So the one line
                    # that says what the retry changed reported the block that
                    # did NOT change and hid the one that did:
                    #
                    #   retrying with {'R0_BLOCK': 128}    six times in a row
                    #
                    # reads as a loop that shrinks nothing, which is the opposite
                    # of what was happening. Measured on Qwen2-MoE's sort kernel.
                    logger.info(
                        "[triton-npu] %s: %d bytes/lane over a budget of %d, "
                        "retrying with %s", kernel_name, over[0], over[1],
                        {k: v for k, v in meta["fixed_config"].items()
                         if k.endswith("BLOCK")})
            # The spec now records the block sizes that actually compiled, and
            # timing.store_meta above wrote the ones that did not. Restate it so
            # a standalone timing run launches the grid the ELF was built for.
            timing.store_meta(write_path, meta)
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
