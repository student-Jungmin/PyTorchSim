"""swiglu — timing-mode validation cases.

size = (rows, interm).  Run: `python timing_mode_validation/ops/swiglu.py --dtype float32`.
"""

PARAMS = "rows, interm"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("swiglu_2048x14336", (2048, 14336), "census", "shape", "llama3-8b act_fn(gate)*up", "MLP의 elementwise 절반"),
    ("swiglu_2048x24576", (2048, 24576), "census", "shape", "gemma-7b", ""),
    ("swiglu_512x14336", (512, 14336), "census", "shape", "llama3-8b", ""),
    # --- added ---
    ("swiglu_2048x11008", (2048, 11008), "added", "none", "interm 격자", "11008/28672은 출하 폭"),
    ("swiglu_2048x16384", (2048, 16384), "added", "shape", "interm 격자", "11008/28672은 출하 폭"),
    ("swiglu_2048x28672", (2048, 28672), "added", "none", "interm 격자", "11008/28672은 출하 폭"),
    ("swiglu_2048x4096", (2048, 4096), "added", "shape", "interm 격자", "11008/28672은 출하 폭"),
    ("swiglu_2048x8192", (2048, 8192), "added", "shape", "interm 격자", "11008/28672은 출하 폭"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
