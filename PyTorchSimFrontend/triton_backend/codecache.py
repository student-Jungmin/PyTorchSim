"""Compile cache for the codegen route: one kernel in, one launcher out.

    define_kernel   ->  triton_npu_compile(src, meta, kernel_name)  ->  launcher
    call site       ->  launcher(arg0, arg1, ..., xnumel)

One directory per source hash, holding the tnpu kernel file and every artifact.
"""

import os
import re

from filelock import FileLock
from torch._inductor.codecache import get_hash

from PyTorchSimFrontend import extension_config

from . import breakdown, functional, kernel_spec, provenance, timing, compiler_bridge

logger = extension_config.setup_logger()

LOCK_TIMEOUT = 600

_SPAD_OVERFLOW_RE = re.compile(r"tnpu-spad-overflow: usage=(\d+) budget=(\d+)")


def _write_path(src_code):
    return os.path.join(extension_config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip())[1:12])


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper.

    Each call is one launch of the whole grid, Spike first so the tensors hold
    real values even if TOGSim fails. Both halves switch on the config keys.
    """
    def __init__(self, kernel_name, workdir, meta):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta

    def __call__(self, *args):
        if extension_config.pytorchsim_functional_mode:
            with breakdown.span(breakdown.SPIKE, self.kernel_name):
                written = functional.run(self.workdir, self.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        if not extension_config.pytorchsim_timing_mode:
            logger.warning(
                "[timing] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        result = timing.run(self.workdir, self.meta, args)
        if isinstance(result, int):
            logger.info("[TOGSim] %s queued as kernel %d; the stream's cycles "
                        "are reported when the simulator closes",
                        self.kernel_name, result)
        else:
            logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


def _spad_overflow(exc):
    """(usage, budget) if this failure was the scratchpad, else None.

    Read off `exc.output`, NOT `str(exc)`: CompilerError's message keeps only lines
    that look like a diagnostic, and this marker is addressed to this function.
    """
    m = _SPAD_OVERFLOW_RE.search(getattr(exc, "output", None) or str(exc))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _shrink_tile(meta, usage, budget):
    """Divide the tile by enough to fit, in place. False if stuck.

    ONLY A BLOCK THE KERNEL TAKES AS AN ARGUMENT MOVES ANYTHING: a persistent
    reduction bakes R0_BLOCK into its body. XBLOCK only when it takes none.
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

    Called from the generated wrapper at module import time. Synchronous, on
    purpose: a thread pool buys nothing until the pipeline itself is proven.
    """
    write_path = _write_path(src_code)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = compiler_bridge.artifact(write_path, "elf")
        if elf is not None and not provenance.matches(write_path):
            logger.info(
                "[triton-npu] %s: cached artifacts carry a different toolchain "
                "or machine identity, rebuilding", kernel_name)
            provenance.clear_stale(write_path)
            elf = None
        if elf is None:
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            timing.store_meta(write_path, meta)
            last_usage = None
            while True:
                kernel_spec.write_spec_file(src_code, meta, spec_path,
                                            compiler_bridge.tnpu_dir())
                try:
                    with breakdown.span(breakdown.TNPU, kernel_name):
                        compiler_bridge.run_pipeline(spec_path, write_path,
                                                 to_stage="binary")
                    breakdown.ingest_tnpu(write_path, kernel_name)
                    break
                except compiler_bridge.CompilerError as exc:
                    over = _spad_overflow(exc)
                    if over is None:
                        raise
                    # A RETRY THAT FREES NOTHING IS NOT A RETRY. The tile
                    # keeps halving, the measurement does not move, and the last
                    # halving takes the outer axis to 1 -- a unit axis, which
                    # leaves select_lane_axis no axis for the lanes, so it
                    # answers differently per buffer and the fold crosses lanes.
                    # That kernel COMPILES and returns wrong numbers.
                    #
                    #   measured, Qwen2-MoE's sort kernel at spad 131072, each
                    #   XBLOCK compiled alone and its .spad read back:
                    #
                    #     128 273,936 | 32 149,264 | 16 132,592   all axis 0
                    #       8 127,616 |  4 127,616 |  2 127,616   all axis 0
                    #       1  37,792                axis 0 x300, 1 x125, ONE_LANE x4
                    #
                    #   8/4/2 are one number because reserve_per_lane rounds up
                    #   to MIN_VEC; 1 is smaller only because the banking
                    #   collapsed. The model gave Max abs diff 1.369991421699524
                    #   at XBLOCK 1, twice, and passes at 8 and 64.
                    #
                    # A compile error names the kernel and the budget; a wrong
                    # number names nothing.
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
            timing.store_meta(write_path, meta)
            provenance.store(write_path)
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
