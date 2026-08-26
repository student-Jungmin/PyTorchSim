"""The timing half of the Triton route: the compiler IR -> trace.so -> TOGSim.

    run(workdir, meta, args)    emit if needed, then simulate, under the lock
    emit_trace(workdir, meta)   *-custom.mlir -> trace.so + trace_cycles.tsv
    run_togsim(workdir, ...)    hand them to TOGSim, return its parsed result
"""

import json
import os
import shutil

from PyTorchSimFrontend import extension_config

from . import breakdown, launch, session

logger = extension_config.setup_logger()

TRACE_SO = "trace.so"
CYCLE_TSV = "trace_cycles.tsv"
SHAPE_TXT = "trace_shape.txt"
META_JSON = "meta.json"

PLACEHOLDER_CYCLE = 1

SAMPLE_MLIR = "04-sample.mlir"
CYCLE_BIN = "cycle_bin"

AXES_TXT = "trace_axes.txt"

LOCK_NAME = ".timing.lock"
LOCK_TIMEOUT = 1800



def measure_tile_cycles(workdir, meta):
    """Per-compute-node cycle counts for ONE tile, measured under gem5.

    The compiler builds the sample binary; None on any failure, and the caller
    then uses the placeholder table.
    """
    from . import compiler_bridge

    kernel_name = meta["kernel_name"]
    spec = os.path.join(workdir, f"{kernel_name}_spec.py")
    if not os.path.isfile(spec):
        logger.warning("[Gem5] %s not found; cannot sample cycles", spec)
        return None

    with breakdown.span(breakdown.GEM5_BUILD, kernel_name):
        rc, output = compiler_bridge.run_module(f"{compiler_bridge.COMPILER_PKG}.trace.emit_gem5_binary", spec, workdir)
    breakdown.ingest_psto(workdir, kernel_name, kind="cycle",
                          name="timing-cycle.json")
    if rc != 0:
        logger.warning("[Gem5] cycle binary build failed:\n%s", output[-2000:])
        return None

    from Simulator.simulator import CycleSimulator
    try:
        with breakdown.span(breakdown.GEM5_RUN, kernel_name):
            return CycleSimulator().compile_and_simulate(
                os.path.join(workdir, CYCLE_BIN),
                int(extension_config.vpu_num_lanes), silent_mode=True)
    except Exception as e:
        logger.warning("[Gem5] sampling failed: %s", e)
        return None


def write_shape(workdir, meta, args=()):
    """Write the grid extents the trace producer reads as shape_args.

    The trailing integers of `args` are the parallel numels, in meta order. A
    template's trailing arguments ARE its grid, so its extents are used as is.
    """
    numels = dict(meta["numels"])
    if meta.get("template_grid") is not None:
        grid = list(launch.launch_extents(meta))
        _write_extents(workdir, grid, "template grid")
        return grid

    passed = [k for k in numels if not k.startswith("r")]
    trailing = [a for a in args if isinstance(a, int) and not isinstance(a, bool)]
    if passed and len(trailing) >= len(passed):
        for key, val in zip(passed, trailing[-len(passed):]):
            numels[key] = val

    grid = list(launch.grid_of({**meta, "numels": numels}))
    _write_extents(workdir, grid, "grid")
    return grid


def _write_extents(workdir, ext, what):
    """Write `ext` as this launch's shape_args, refusing a count it cannot use.

    The count is checked against the shared trace; the file goes in this
    process's launch directory, because the grid belongs to the launch.
    """
    path = os.path.join(workdir, AXES_TXT)
    if os.path.isfile(path):
        with open(path) as f:
            want = int(f.read().strip())
        if want != len(ext):
            raise ValueError(
                f"{TRACE_SO} in {workdir} was compiled to read {want} grid "
                f"extent(s) and this launch has {len(ext)} ({ext}); the trace "
                f"would read past the end of shape_args. Delete the workdir to "
                f"rebuild it.")
    with open(os.path.join(session.dir_for(workdir), SHAPE_TXT), "w") as f:
        f.write("\n".join(str(int(g)) for g in ext) + "\n")
    logger.info("[TOGSim] %s %s -> %s", what, ext, SHAPE_TXT)


def emit_trace(workdir, meta):
    """Build trace.so + trace_cycles.tsv from the compiler's trace producer.

    Returns the number of compute tiles the cycle table covers.
    """
    import json

    from . import compiler_bridge, trace_build
    from .compiler_bridge import artifact

    kernel = meta["kernel_name"]
    spec = os.path.join(workdir, f"{kernel}_spec.py")
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"{spec} not found; cannot build the trace")

    with breakdown.span(breakdown.GEM5_BUILD, kernel):
        rc, output = compiler_bridge.run_module(
            f"{compiler_bridge.COMPILER_PKG}.trace.emit_togsim_so", spec, workdir)
    if rc != 0:
        raise RuntimeError(f"the compiler could not emit a trace producer for "
                           f"{kernel}:\n{output[-2000:]}")

    so_path = artifact(workdir, "trace_so")
    types_path = artifact(workdir, "tile_types")
    if so_path is None or types_path is None:
        raise FileNotFoundError(
            f"{workdir} has no trace_so/tile_types -- the compiler must reach "
            f"the trace step, which is what builds the producer")
    with open(types_path) as fh:
        tiles = json.load(fh)
    compute_types = tiles["compute_types"]
    axes = tiles["parallel_axes"]
    n_tiles = len(compute_types)

    cycles = measure_tile_cycles(workdir, meta)
    if cycles is None:
        raise RuntimeError(
            f"[Gem5] sampling failed for {kernel}, so compute latency would "
            f"not be modelled at all")
    lanes = int(extension_config.vpu_num_lanes)
    table = trace_build.cycle_table(compute_types, list(cycles),
                                    x_offset=lanes, w_offset=0)

    if os.path.abspath(so_path) != os.path.abspath(os.path.join(workdir, TRACE_SO)):
        shutil.copyfile(so_path, os.path.join(workdir, TRACE_SO))
    with open(os.path.join(workdir, AXES_TXT), "w") as f:
        f.write(f"{len(axes)}\n")

    trace_build.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    logger.info("[Gem5] tile cycles: %s", table)
    return n_tiles


def run_togsim(workdir, meta, args=()):
    """Simulate the emitted trace, or hand it to an open TOGSimulator.

    Standalone returns the parsed result; a queued launch returns its kernel id,
    since the stream's numbers are only whole when the simulator closes.
    """
    import torch

    from Simulator.simulator import TOGSimulator

    so = os.path.join(workdir, TRACE_SO)
    if not os.path.isfile(so):
        raise FileNotFoundError(f"{so} not found -- call emit_trace first")
    mine = session.link_shared(workdir, (TRACE_SO, CYCLE_TSV))
    write_shape(workdir, meta, args)

    handle = os.path.join(mine, "tile_graph.onnx")
    attribute = os.path.join(mine, "attribute")
    if torch.npu.get_tog_simulator() is not None:
        return torch.npu.launch_kernel(handle, attribute)

    result_path = TOGSimulator.run_standalone(handle, attribute)
    return TOGSimulator.get_result_from_file(result_path)


def run(workdir, meta, args=()):
    """Emit the trace if it is missing, then simulate. Returns TOGSim's result.

    Only the build is locked, and only against a second build of the same
    trace; simulating is per launch and runs concurrently.
    """
    from filelock import FileLock

    kernel = meta["kernel_name"]
    if not os.path.isfile(os.path.join(workdir, TRACE_SO)):
        with FileLock(os.path.join(workdir, LOCK_NAME), timeout=LOCK_TIMEOUT):
            if not os.path.isfile(os.path.join(workdir, TRACE_SO)):
                with breakdown.span(breakdown.TOGSIM_TRACE, kernel):
                    emit_trace(workdir, meta)
    with breakdown.span(breakdown.TOGSIM_RUN, kernel):
        return run_togsim(workdir, meta, args)


def store_meta(workdir, meta):
    """Persist codegen metadata beside the artifacts, so the timing step can
    run standalone."""
    with open(os.path.join(workdir, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)
