"""mlp_swiglu — timing-mode validation cases.

size = (S, hidden, interm).  Run: `python timing_mode_validation/ops/mlp_swiglu.py --dtype float32`.
"""

PARAMS = "S, hidden, interm"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("mlp_swiglu_2048x4096x14336", (2048, 4096, 14336), "census", "none", "llama3-8b MLP 블록", "GEMM 3개+elementwise"),
    ("mlp_swiglu_512x4096x14336", (512, 4096, 14336), "census", "none", "llama3-8b MLP 블록", ""),
    # --- added ---
    ("mlp_swiglu_128x4096x14336", (128, 4096, 14336), "added", "none", "블록 seq 격자", ""),
    ("mlp_swiglu_4096x4096x14336", (4096, 4096, 14336), "added", "none", "블록 seq 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
