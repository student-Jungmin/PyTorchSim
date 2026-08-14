"""Tile-Operation-Graph generation from the MLIR that triton-npu emits.

    build_tog        post-vcix MLIR -> TogBuilder nodes (and the sample rewrite)
    build_skeleton   those nodes -> a per-tile skeleton with dependencies
    cycle_table      the skeleton + gem5 cycles -> trace_cycles.tsv
    lower_to_emitc   the skeleton -> the C++ trace producer, built as trace.so

TOGSim turns trace.so plus the cycle table into a TileGraph. Nothing here reads
PyTorchSim state: the input is a .mlir file and the output is on disk.
"""
def _ensure_mlir_bindings_on_path():
    """Make `import mlir` work even when PYTHONPATH is not set, by deriving the
    bindings location from TORCHSIM_LLVM_PATH (e.g. /riscv-llvm/bin ->
    /riscv-llvm/python_packages/mlir_core). The container sets PYTHONPATH, but
    plain local runs may not."""
    try:
        import mlir.ir  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import os
    import sys
    from PyTorchSimFrontend import extension_config
    llvm_path = (extension_config.CONFIG_TORCHSIM_LLVM_PATH or "").rstrip("/")
    cand = os.path.join(os.path.dirname(llvm_path), "python_packages", "mlir_core")
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)


_ensure_mlir_bindings_on_path()

from .build_tog import run_tog  # noqa: F401,E402 (re-exported)
