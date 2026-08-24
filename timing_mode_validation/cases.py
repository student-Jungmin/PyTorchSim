"""Load the per-operation case files in `ops/` as one list.

Each `ops/<op>.py` owns its operation's cases; nothing else defines a case.
"""

import glob
import importlib.util
import os

OPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")

FIELDS = ("name", "size", "origin", "census", "source", "note")


def load_op(op):
    """The module for one operation, imported by path."""
    path = op if op.endswith(".py") else os.path.join(OPS_DIR, f"{op}.py")
    spec = importlib.util.spec_from_file_location(
        f"tmv_case_{os.path.basename(path)[:-3]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OP = os.path.basename(path)[:-3]
    return mod


def ops():
    """Every operation that has a case file, in name order."""
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(OPS_DIR, "*.py")))


def load(op_files=None, origin=None):
    """Cases as dicts, optionally filtered by where the shape came from."""
    out = []
    for op in (op_files or ops()):
        mod = load_op(op)
        for case in mod.CASES:
            row = dict(zip(FIELDS, case), op=mod.OP, params=mod.PARAMS)
            if origin and row["origin"] not in origin:
                continue
            out.append(row)
    return out
