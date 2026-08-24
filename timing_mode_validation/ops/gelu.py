"""gelu — timing-mode validation cases.

size = (rows, interm).  Run: `python timing_mode_validation/ops/gelu.py --dtype float32`.
"""

PARAMS = "rows, interm"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("gelu_2048x3072", (2048, 3072), "census", "shape", "gpt2 mlp", ""),
    ("gelu_512x3072", (512, 3072), "census", "shape", "bert-base intermediate", ""),
    # --- added ---
    ("gelu_2048x12288", (2048, 12288), "added", "shape", "interm 격자", ""),
    ("gelu_2048x4096", (2048, 4096), "added", "shape", "interm 격자", ""),
    ("gelu_2048x8192", (2048, 8192), "added", "shape", "interm 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
