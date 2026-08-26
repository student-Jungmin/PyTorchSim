"""Tile-Operation-Graph generation from the MLIR that pytorchsim-triton-opt emits.

    build_tog        post-vcix MLIR -> TogBuilder nodes (and the sample rewrite)
    build_skeleton   those nodes -> a per-tile skeleton with dependencies
    cycle_table      the skeleton + gem5 cycles -> trace_cycles.tsv
    lower_to_emitc   the skeleton -> the C++ trace producer, built as trace.so

TOGSim turns trace.so plus the cycle table into a TileGraph. Nothing here reads
PyTorchSim state: the input is a .mlir file and the output is on disk.

Importing this package SELECTS the MLIR bindings beside TORCHSIM_LLVM_PATH and
drops every other LLVM's from sys.path. `mlir` is a namespace package, so two
LLVMs in one interpreter merge silently instead of failing.
"""
import os
import re
import sys

#: The LLVM this package is written against. The compiler prints the IR parsed
#: here, so a different one is not a smaller machine but a different reader.
REQUIRED_LLVM_MAJOR = 23


def llvm_major(bin_dir=None):
    """LLVM_VERSION_MAJOR of the install holding `bin_dir`, or None if unreadable."""
    from PyTorchSimFrontend import extension_config
    if bin_dir is None:
        bin_dir = extension_config.CONFIG_TORCHSIM_LLVM_PATH or ""
    header = os.path.join(os.path.dirname(bin_dir.rstrip("/")),
                          "include", "llvm", "Config", "llvm-config.h")
    try:
        with open(header) as fh:
            m = re.search(r"define\s+LLVM_VERSION_MAJOR\s+(\d+)", fh.read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def check_llvm_version():
    """Raise if TORCHSIM_LLVM_PATH is not REQUIRED_LLVM_MAJOR. Silent if unreadable."""
    from PyTorchSimFrontend import extension_config
    got = llvm_major()
    if got is None or got == REQUIRED_LLVM_MAJOR:
        return
    raise RuntimeError(
        f"TORCHSIM_LLVM_PATH is LLVM {got}, and this package is written against "
        f"LLVM {REQUIRED_LLVM_MAJOR} -- the one the compiler prints its IR with. "
        f"A wrong reader does not fail at the seam: it fails deep inside "
        f"build_tog or lower_to_emitc with an IndexError or an unknown emitc "
        f"type, which reads as a bug in this repository.\n"
        f"  configured: {extension_config.CONFIG_TORCHSIM_LLVM_PATH}\n"
        f"  point TORCHSIM_LLVM_PATH at the LLVM {REQUIRED_LLVM_MAJOR} install "
        f"(`source /workspace/psto-env.sh`, or run the torchsim_psto_base image), "
        f"and `pytorchsim-triton-opt doctor` reports what both sides hold.")


def bindings_dir():
    """The mlir_core beside TORCHSIM_LLVM_PATH (`/x/bin` -> `/x/python_packages/mlir_core`)."""
    from PyTorchSimFrontend import extension_config
    llvm = (extension_config.CONFIG_TORCHSIM_LLVM_PATH or "").rstrip("/")
    return os.path.join(os.path.dirname(llvm), "python_packages", "mlir_core")


def activate_bindings(sys_path=None):
    """Put `bindings_dir()` first on the path and REMOVE foreign mlir roots.

    Returns the roots removed; out-ranking them is not enough for a namespace
    package, whose `__path__` is the union of every `mlir/` on the path.
    """
    sys_path = sys.path if sys_path is None else sys_path
    keep = bindings_dir()
    real = os.path.realpath(keep)
    foreign = [p for p in list(sys_path)
               if p and os.path.realpath(p) != real
               and os.path.isdir(os.path.join(p, "mlir", "_mlir_libs"))]
    for p in foreign:
        sys_path.remove(p)
    if os.path.isdir(keep):
        if keep in sys_path:
            sys_path.remove(keep)
        sys_path.insert(0, keep)
    return foreign


def loaded_bindings():
    """The files the loaded `mlir*` modules came from, [] if none is loaded.

    Read off sys.modules, NOT `mlir.__path__`: the latter recomputes from
    sys.path, so a submodule imported before `activate_bindings` stays.
    """
    seen = {getattr(m, "__file__", None)
            for n, m in list(sys.modules.items())
            if n == "mlir" or n.startswith("mlir.")}
    return sorted(f for f in seen if f)


def check_bindings():
    """Raise if `mlir` resolved outside `bindings_dir()`. Silent when it agrees."""
    keep = os.path.realpath(bindings_dir()) + os.sep
    stray = sorted({p for p in loaded_bindings()
                    if not os.path.realpath(p).startswith(keep)})
    if stray:
        raise RuntimeError(
            f"the MLIR python bindings loaded from {stray}, not the configured "
            f"{bindings_dir()}. the compiler prints this IR with its own LLVM and a "
            f"different one reading it back is the seam that stopped being "
            f"papered over; point TORCHSIM_LLVM_PATH at the LLVM the compiler uses.")


activate_bindings()
check_llvm_version()

from .build_tog import run_tog  # noqa: F401,E402 (re-exported)

check_bindings()
