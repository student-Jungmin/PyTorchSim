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

        @triton_heuristics.persistent_reduction(size_hints={'x': 1, 'r0_': 64})
        def triton_npu_fused_copy__sort_40(..., XBLOCK : tl.constexpr):
            R0_BLOCK: tl.constexpr = 64         # <- not an argument

    so R0_BLOCK is absent from the signature and rewriting it here changes the
    spec, recompiles, and produces THE SAME KERNEL.

        measured   Qwen3's fused q_norm + rope, 27 scratchpad globals over a
                   [XBLOCK, 128] tile (develop-e2e-qwen f8deeb3, where this
                   was first diagnosed):
                     R0_BLOCK 128 -> 8 -> 1  1625120 bytes/lane, all three
                     XBLOCK   128 -> 64      fits

        measured   Moonlight's MoE router sort at 16 experts, xnumel 1,
                   r0_numel 64: R0_BLOCK 32 and 16 both 104224 bytes/lane.
                   Same shape, and the same lever is the one that moves.

    Hence: reduction blocks if the kernel takes any, XBLOCK only when it takes
    none, and never both.
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
            # THE CORRECTION IS ONLY A CORRECTION IF THE NUMBER MOVES. Halving
            # the reduction block is a guess about what the kernel keeps live,
            # and for a kernel whose scratchpad is sized by something else the
            # remeasured usage comes back IDENTICAL -- so the loop walks
            # R0_BLOCK down to 1, paying one full recompile per step, and fails
            # for the reason it already knew at the first retry.
            #
            #     measured   Moonlight at 16 experts, `sort_40` (the MoE
            #                router): 104224 bytes/lane at R0_BLOCK 32 and
            #                104224 again at 16, each measurement about eight
            #                minutes because the sort's IR is 277 KB. Five more
            #                steps were queued behind it.
            #
            # So a retry that does not reduce the usage ends it: the block size
            # is not this kernel's free variable, and the caller gets the spad
            # overflow with the kernel's name on it instead of forty minutes
            # later.
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
                    if last_usage is not None and over[0] >= last_usage:
                        logger.info(
                            "[triton-npu] %s: %d bytes/lane did not move when "
                            "the reduction block was halved, so the block is "
                            "not what sizes it -- not retrying", kernel_name,
                            over[0])
                        raise
                    if not _shrink_tile(meta, *over):
                        raise
                    last_usage = over[0]
                    logger.info(
                        "[triton-npu] %s: %d bytes/lane over a budget of %d, "
                        "retrying with %s", kernel_name, over[0], over[1],
                        # `XBLOCK` HAS NO UNDERSCORE, so `endswith("_BLOCK")`
                        # printed every block EXCEPT the one _shrink_tile had
                        # just moved -- and the unchanged R0_BLOCK next to it
                        # read as "the retry did nothing". Measured on the
                        # router sort: the line said {'R0_BLOCK': 64} while the
                        # spec said XBLOCK 128 -> 64.
                        {k: v for k, v in meta["fixed_config"].items()
                         if k.endswith("BLOCK")})
            # The spec now records the block sizes that actually compiled, and
            # timing.store_meta above wrote the ones that did not. Restate it so
            # a standalone timing run launches the grid the ELF was built for.
            timing.store_meta(write_path, meta)
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
