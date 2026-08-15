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

from . import breakdown, functional, kernel_spec, timing, tnpu_bridge

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
        logger.info("[TOGSim] %s simulated -> %s", self.kernel_name, result)
        return result


def _spad_overflow(exc):
    """(usage, budget) if this failure was the scratchpad, else None.

    Read off `exc.output`, NOT `str(exc)`: TnpuError's message keeps only lines
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
        elf = tnpu_bridge.stage_artifact(write_path, f"{kernel_name}.elf")
        if elf is None:
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            timing.store_meta(write_path, meta)
            while True:
                kernel_spec.write_spec_file(src_code, meta, spec_path,
                                            tnpu_bridge.tnpu_dir())
                try:
                    with breakdown.span(breakdown.TNPU, kernel_name):
                        tnpu_bridge.run_pipeline(spec_path, write_path,
                                                 to_stage="binary")
                    breakdown.ingest_tnpu(write_path, kernel_name)
                    break
                except tnpu_bridge.TnpuError as exc:
                    over = _spad_overflow(exc)
                    if over is None or not _shrink_tile(meta, *over):
                        raise
                    logger.info(
                        "[triton-npu] %s: %d bytes/lane over a budget of %d, "
                        "retrying with %s", kernel_name, over[0], over[1],
                        {k: v for k, v in meta["fixed_config"].items()
                         if k.endswith("_BLOCK")})
            timing.store_meta(write_path, meta)
        logger.info("[triton-npu] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta)
