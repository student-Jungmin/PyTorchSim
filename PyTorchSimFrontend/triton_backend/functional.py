"""The functional half of the Triton route: real tensors -> Spike -> real tensors.

    run(workdir, meta, args)    args -> runtime/*.raw -> spike -> args

The binary is shape-specialised, so a launch whose shapes differ from the
compiled ones is rejected rather than run against the wrong bounds.
"""

import os

from PyTorchSimFrontend import extension_config

from . import breakdown, layout, session

logger = extension_config.setup_logger()

REPLAY_DIR = ".triton_replay"


class ShapeMismatch(RuntimeError):
    """The launch does not match the shapes the binary was compiled for."""


def _np_dtype(name):
    import numpy as np
    return np.dtype("bool" if name == "bool" else name)


def tensor_args(meta, args):
    """[(arg_meta, tensor)] for the launch, paired by position.

    Inductor passes the tensors first and the numels after, in signature order,
    so `meta["args"]` lines up with the leading arguments.
    """
    import torch

    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    metas = meta["args"]
    if len(tensors) != len(metas):
        raise ShapeMismatch(
            f"{meta['kernel_name']}: launch passed {len(tensors)} tensor(s), "
            f"but the spec declares {len(metas)} ({[m['name'] for m in metas]})")
    return list(zip(metas, tensors))


def _storage_numel(t):
    """How many elements of storage `t` spans -- its last addressable one.

    NOT `t.numel()`: the .raw round trip is a flat file and has to be as long
    as the addresses the kernel computes. Same definition as the compile side.
    """
    return layout.storage_span(t.shape, t.stride())


def _check(meta, pairs):
    for m, t in pairs:
        if _storage_numel(t) != m["numel"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' spans "
                f"{_storage_numel(t)} element(s) of storage, but the binary was "
                f"compiled for {m['numel']}. "
                f"the compiler bakes the extents, the grid and the scalar values into "
                f"the kernel, so a dynamic-shape graph reuses an ELF that does "
                f"not fit. The timing path does handle this (it takes the grid "
                f"at run time); set pytorchsim_functional_mode: False to study "
                f"cycles alone, or keep shapes static to check values.")
        if str(t.dtype).removeprefix("torch.") != m["dtype"]:
            raise ShapeMismatch(
                f"{meta['kernel_name']}: '{m['name']}' is {t.dtype}, but the "
                f"binary was compiled for {m['dtype']}")


def write_inputs(workdir, meta, args):
    """Write every arg as runtime/<name>.raw. Returns the runtime directory.

    Inputs are scattered to the addresses the kernel will compute; outputs are
    zeros, because the wrapper dumps by argv position and a gap shifts them.
    """
    import numpy as np
    import torch

    pairs = tensor_args(meta, args)
    _check(meta, pairs)

    runtime = session.runtime_dir(workdir)
    for m, t in pairs:
        path = os.path.join(runtime, f"{m['name']}.raw")
        if m["role"] not in ("in", "inout"):
            np.zeros(m["numel"], dtype=_np_dtype(m["dtype"])).tofile(path)
            continue
        cpu = t.detach().to("cpu")
        flat = torch.zeros(m["numel"], dtype=cpu.dtype, device="cpu")
        shape, strides = list(t.shape), list(t.stride())
        src = cpu
        for i in range(len(strides) - 1, -1, -1):
            if strides[i] == 0:
                src = src.select(i, 0)
                shape.pop(i)
                strides.pop(i)
        flat.as_strided(shape, strides).copy_(src)
        flat.numpy().tofile(path)
    return runtime


def read_outputs(workdir, meta, args):
    """Copy the .raw files Spike wrote back into the launch's output tensors.

    Gathered back the way they were scattered: the kernel wrote the addresses
    the strides name, so `view_as` would be the same defect as `.contiguous()`.
    """
    import numpy as np
    import torch

    runtime = session.runtime_dir(workdir)
    written = []
    for m, t in tensor_args(meta, args):
        if m["role"] not in ("out", "inout"):
            continue
        path = os.path.join(runtime, f"{m['name']}.raw")
        flat = np.fromfile(path, dtype=_np_dtype(m["dtype"]))
        if flat.size != m["numel"]:
            raise RuntimeError(
                f"{path} holds {flat.size} element(s), expected {m['numel']} "
                f"-- Spike did not write the whole tensor")
        stored = torch.from_numpy(flat).as_strided(t.shape, t.stride())
        t.copy_(stored.to(t.dtype))
        written.append(m["name"])
    return written


def _replay_root(workdir):
    """Beside the workdirs, not inside one.

    A the compiler-side fix is picked up by DELETING `outputs/triton_*`, and a cache
    kept inside a workdir would go with it every time it was most wanted.
    """
    return os.path.join(os.path.dirname(os.path.abspath(workdir)), REPLAY_DIR)


def _replay_key(workdir, meta, runtime):
    """What this launch's outputs are a function of.

    The ELF's own bytes, so a fix anywhere in the compiler misses, and the bytes of
    every input. The Triton source is already in the workdir path.
    """
    import hashlib

    h = hashlib.sha256()
    elf = [f for f in sorted(os.listdir(workdir)) if f.endswith(".elf")]
    if not elf:
        return None
    with open(os.path.join(workdir, elf[0]), "rb") as f:
        h.update(f.read())
    for m in meta["args"]:
        h.update(("%s|%s|%s|%s;" % (m["name"], m["role"], m["dtype"],
                                    m["numel"])).encode())
        if m["role"] in ("in", "inout"):
            with open(os.path.join(runtime, f"{m['name']}.raw"), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:32]


def _outputs_of(meta):
    return [m["name"] for m in meta["args"] if m["role"] in ("out", "inout")]


def _replay(workdir, meta, runtime, key):
    """Put a saved run's outputs back in runtime/, or say it is not there."""
    import shutil

    saved = os.path.join(_replay_root(workdir), key)
    names = _outputs_of(meta)
    if not all(os.path.isfile(os.path.join(saved, f"{n}.raw")) for n in names):
        return False
    for n in names:
        shutil.copyfile(os.path.join(saved, f"{n}.raw"),
                        os.path.join(runtime, f"{n}.raw"))
    return True


def _save_replay(workdir, meta, runtime, key):
    import shutil

    saved = os.path.join(_replay_root(workdir), key)
    os.makedirs(saved, exist_ok=True)
    for n in _outputs_of(meta):
        shutil.copyfile(os.path.join(runtime, f"{n}.raw"),
                        os.path.join(saved, f"{n}.raw"))


def run(workdir, meta, args):
    """Execute the kernel on the launch's tensors. Returns the names written.

    No lock: every file this writes is under `session.runtime_dir`, which is
    this process's own, so concurrent launches of one kernel never meet.
    Replay is OFF by default -- a result out of a file is not one the simulator
    produced today; it is for the inner loop, not for reporting.
    """
    from . import compiler_bridge

    spec = os.path.join(workdir, f"{meta['kernel_name']}_spec.py")
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"{spec} not found -- compile the kernel first")

    runtime = write_inputs(workdir, meta, args)

    key = None
    if os.environ.get("TORCHSIM_TRITON_REPLAY", "0") == "1":
        key = _replay_key(workdir, meta, runtime)
        if key and _replay(workdir, meta, runtime, key):
            logger.info("[Spike] %s replayed %s (same ELF, same inputs)",
                        meta["kernel_name"], key)
            return read_outputs(workdir, meta, args)

    rc, output = compiler_bridge.run_module(f"{compiler_bridge.COMPILER_PKG}.spike", spec, workdir,
                                        "--runtime", runtime)
    breakdown.ingest_psto(runtime, meta["kernel_name"], kind="spike",
                          name="timing-spike.json")
    if rc != 0:
        raise RuntimeError(
            f"[Spike] {meta['kernel_name']} failed:\n" + output[-2000:])

    if key:
        _save_replay(workdir, meta, runtime, key)
    return read_outputs(workdir, meta, args)
