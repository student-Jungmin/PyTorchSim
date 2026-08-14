"""The timing half of the Triton route: tnpu IR -> trace.so -> TOGSim.

    emit_trace(workdir, meta)   *-custom.mlir -> trace.so + trace_cycles.tsv
    run_togsim(workdir, ...)    hand them to TOGSim, return its parsed result
"""

import json
import os

from PyTorchSimFrontend import extension_config

from . import launch

logger = extension_config.setup_logger()

TRACE_SO = "trace.so"
CYCLE_TSV = "trace_cycles.tsv"
SHAPE_TXT = "trace_shape.txt"
META_JSON = "meta.json"

PLACEHOLDER_CYCLE = 1

SAMPLE_MLIR = "04-sample.mlir"
CYCLE_BIN = "cycle_bin"

AXES_TXT = "trace_axes.txt"

_PID_SLOT = {"x": 0, "y": 1, "z": 2}


def measure_tile_cycles(workdir, meta):
    """Per-compute-node cycle counts for ONE tile, measured under gem5.

    build_tog's sample mode makes every loop a single trip, tnpu lowers it and
    gem5 runs it. None on any failure; the caller uses the placeholder table.
    """
    from PyTorchSimFrontend.tog.build_tog import run_tog

    from . import tnpu_bridge
    from .tnpu_bridge import stage_artifact

    kernel_name = meta["kernel_name"]
    spec = os.path.join(workdir, f"{kernel_name}_spec.py")
    if not os.path.isfile(spec):
        logger.warning("[Gem5] %s not found; cannot sample cycles", spec)
        return None

    run_tog(stage_artifact(workdir, "custom.mlir"),
            os.path.join(workdir, "tog_sample.py"),
            os.path.join(workdir, SAMPLE_MLIR), sample_mode=True)

    rc, output = tnpu_bridge.run_module("tnpu.cycle", spec, workdir)
    if rc != 0:
        logger.warning("[Gem5] cycle binary build failed:\n%s", output[-2000:])
        return None

    from Simulator.simulator import CycleSimulator
    try:
        return CycleSimulator().compile_and_simulate(
            os.path.join(workdir, CYCLE_BIN), int(extension_config.vpu_num_lanes),
            silent_mode=True)
    except Exception as e:
        logger.warning("[Gem5] sampling failed: %s", e)
        return None


def _runtime_arg_layout(meta):
    """(n_tensor_args, n_scalar_args) of the lowered signature.

    triton-shared lays it out as pointers, user scalars, then its own six
    (gridX,Y,Z / pidX,Y,Z). constexpr params never become arguments.
    """
    sig = meta["signature"]
    tensors = [k for k, v in sig.items() if v.startswith("*")]
    scalars = [k for k, v in sig.items()
               if not v.startswith("*") and v != "constexpr"]
    return len(tensors), len(scalars)


def work_item_for(meta):
    """The WorkItem describing this kernel's program-id args and grid extents.

    A TEMPLATE'S AXIS COUNT IS NOT IN ITS NUMELS: counting from them gives 1 and
    leaves a pid argument unaccounted. Only the COUNT is compiled in.
    """
    from PyTorchSimFrontend.tog.lower_to_emitc import WorkItem

    n_tensor, n_scalar = _runtime_arg_layout(meta)
    pid_base = n_tensor + n_scalar + 3
    axes = launch.launch_axes(meta)
    return WorkItem(parallel_args=[pid_base + _PID_SLOT[p] for p in axes],
                    grid=[None] * len(axes))


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
    """Write `ext` as shape_args, refusing a count the trace cannot use.

    A trace.so is REUSED from disk, so a stale one can meet a meta that counts
    differently -- and the producer ignores `n`, so that is an allocation.
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
    with open(os.path.join(workdir, SHAPE_TXT), "w") as f:
        f.write("\n".join(str(int(g)) for g in ext) + "\n")
    logger.info("[TOGSim] %s %s -> %s", what, ext, SHAPE_TXT)


def emit_trace(workdir, meta):
    """Build `trace.so` + `trace_cycles.tsv` from tnpu's post-vcix IR.

    Returns the number of compute tiles the cycle table covers.
    """
    from PyTorchSimFrontend.tog import build_skeleton as bs
    from PyTorchSimFrontend.tog import cycle_table as ct
    from PyTorchSimFrontend.tog import lower_to_emitc as l2e
    from PyTorchSimFrontend.tog.build_tog import ir

    from .tnpu_bridge import stage_artifact
    postvcix = stage_artifact(workdir, "custom.mlir")
    if postvcix is None:
        raise FileNotFoundError(
            f"no *-custom.mlir in {workdir} -- tnpu must run far enough to emit "
            f"the post-vcix IR, which is what the trace is built from")

    cycles = measure_tile_cycles(workdir, meta)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(postvcix).read(), ctx)
        bs.build_skeleton(module)
        compute_types = ct._compute_types(module)
        n_tiles = len(compute_types)

        if cycles:
            cl = list(cycles)
            if len(cl) != n_tiles:
                logger.warning("[Gem5] returned %d cycle(s) for %d "
                               "tile(s); padding with the last", len(cl), n_tiles)
                cl = (cl + [cl[-1]] * n_tiles)[:n_tiles]
            lanes = int(extension_config.vpu_num_lanes)
            table = ct.build_cycle_table(module, cl, x_offset=lanes, w_offset=0)
        else:
            table = [(PLACEHOLDER_CYCLE, 0)] * n_tiles
            logger.warning(
                "[Gem5] %s holds PLACEHOLDER cycles (%d per tile x %d "
                "tiles): gem5 sampling did not produce a measurement, so "
                "compute latency is NOT modelled",
                CYCLE_TSV, PLACEHOLDER_CYCLE, n_tiles)

        wi = work_item_for(meta)
        l2e.skeleton_to_so(module, os.path.join(workdir, TRACE_SO), work_item=wi)
    with open(os.path.join(workdir, AXES_TXT), "w") as f:
        f.write(f"{len(wi.parallel_args)}\n")

    ct.dump_cycle_table_tsv(table, os.path.join(workdir, CYCLE_TSV))
    if cycles:
        logger.info("[Gem5] tile cycles: %s", table)
    return n_tiles


def run_togsim(workdir, meta, args=()):
    """Simulate the emitted trace. Returns TOGSimulator's parsed result dict.

    `meta`/`args` supply the grid: the trace producer takes its loop bounds
    from shape_args, so they are written per launch rather than compiled in.
    """
    from Simulator.simulator import TOGSimulator

    so = os.path.join(workdir, TRACE_SO)
    if not os.path.isfile(so):
        raise FileNotFoundError(f"{so} not found -- call emit_trace first")
    write_shape(workdir, meta, args)

    handle = os.path.join(workdir, "tile_graph.onnx")
    result_path = TOGSimulator.run_standalone(
        handle, os.path.join(workdir, "attribute"))
    return TOGSimulator.get_result_from_file(result_path)


def store_meta(workdir, meta):
    """Persist codegen metadata beside the artifacts, so the timing step can
    run standalone."""
    with open(os.path.join(workdir, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)
