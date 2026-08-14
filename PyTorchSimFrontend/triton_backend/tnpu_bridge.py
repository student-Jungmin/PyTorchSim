"""Run the triton-npu pipeline, out of process.

tnpu's passes run on LLVM 23's MLIR python bindings and this process holds LLVM
20's. `mlir` is a NAMESPACE package, so two LLVMs in one interpreter merge
silently; the seam between them is a file, which is measured to work.
"""

import os
import re
import subprocess

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()


class TnpuError(RuntimeError):
    """A tnpu stage failed. Inductor reports only str(exc), so the stage's own
    diagnostic has to travel in the message."""
    _SIGNAL = re.compile(
        r"^(?!\s|Traceback|During handling|The above)"
        r"(.*\berror:\s.*|.*failed to legalize.*|"
        r"[\w.]*(?:Error|Exception)\b.*|.*Assertion.*)$", re.M)
    _FRAME = re.compile(r'^\s|^\s*File "|^\s*\^')

    def __init__(self, message, cmd=None, output=None):
        self.cmd = cmd
        self.output = output
        if output:
            hits = [h.strip() for h in self._SIGNAL.findall(output)
                    if not self._FRAME.match(h)]
            if not hits:
                hits = [l for l in output.strip().splitlines()
                        if l.strip() and not self._FRAME.match(l)]
            message = message + "\n  " + "\n  ".join(l[:300] for l in hits[-3:])
        super().__init__(message)


def tnpu_dir():
    d = extension_config.CONFIG_TNPU_DIR
    if not os.path.isdir(d):
        raise TnpuError(
            f"triton-npu checkout not found at {d}. It is a separate repository "
            f"and is not vendored; clone it there or set TNPU_DIR.")
    return d


def machine():
    """The machine the kernel is compiled for, FROM THE TOGSIM YAML.

    The YAML is the hardware description and therefore the authority; tnpu's
    config.py holds defaults for the same numbers, and the two have drifted.
    """
    spad = extension_config.CONFIG_SPAD_INFO["spad_size"]
    return {"lanes": int(extension_config.vpu_num_lanes),
            "vlen_bits": int(extension_config.vpu_vector_length_bits),
            "spad_size": int(spad)}


def tnpu_env():
    """The environment for a tnpu subprocess: this machine, and no PYTHONPATH.

    The three TNPU_* names are what tnpu/config.py reads, so this tells it rather
    than overrides it; anything already in the environment wins.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    m = machine()
    env.setdefault("TNPU_VECTORLANE_SIZE", str(m["lanes"]))
    env.setdefault("TNPU_VLEN_BITS", str(m["vlen_bits"]))
    env.setdefault("TNPU_SPAD_SIZE", str(m["spad_size"]))
    return env


def doctor():
    """Return (ok, output) for tnpu's own toolchain check."""
    proc = subprocess.run(
        [extension_config.CONFIG_TNPU_PYTHON,
         os.path.join(tnpu_dir(), "run.py"), "doctor"],
        capture_output=True, text=True, cwd=tnpu_dir())
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_module(module, *args, timeout=None):
    """Run `python -m <module> <args>` inside tnpu's checkout. (rc, output).

    Reaching tnpu means tnpu's interpreter, tnpu's cwd and `tnpu_env` -- one
    fact. What to do when it fails differs per caller and stays with them.
    """
    proc = subprocess.run(
        [extension_config.CONFIG_TNPU_PYTHON, "-m", module, *args],
        capture_output=True, text=True, cwd=tnpu_dir(), env=tnpu_env(),
        timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def run_pipeline(spec_path, workdir, to_stage="binary", timeout=1800):
    """Drive tnpu's stages over `spec_path`, writing artifacts into `workdir`.

    Stops at `to_stage`, by default `binary`: stages 6 and 7 want tensors and a
    per-kernel reference this route has no graph-level answer for.
    """
    cmd = [extension_config.CONFIG_TNPU_PYTHON,
           os.path.join(tnpu_dir(), "run.py"), spec_path,
           "--from", "ttir", "--to", to_stage, "--workdir", workdir]

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=tnpu_dir(), env=tnpu_env(), timeout=timeout)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        log = os.path.join(workdir, "stage.log")
        if os.path.isfile(log):
            with open(log, errors="replace") as fh:
                output += "\n" + fh.read()
        raise TnpuError(f"tnpu pipeline failed (exit {proc.returncode})",
                        cmd=" ".join(cmd), output=output)
    logger.debug("[triton-npu] %s", output)
    return workdir


def stage_artifact(workdir, suffix):
    """The stage file ending in `suffix`, whatever number tnpu gave it.

    THE NUMBERS ARE NOT AN INTERFACE -- tnpu renumbers when a stage is added,
    and the post-vcix IR has already moved from 04- to 05-. None if no match.
    """
    import glob
    hits = sorted(glob.glob(os.path.join(workdir, f"*-{suffix}")))
    return hits[-1] if hits else None
