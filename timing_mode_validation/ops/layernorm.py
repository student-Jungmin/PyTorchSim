"""layernorm — timing-mode validation cases.

size = (rows, hidden).  Run: `python timing_mode_validation/ops/layernorm.py --dtype float32`.
"""

PARAMS = "rows, hidden"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("layernorm_2048x4096", (2048, 4096), "census", "none", "bert 폭을 llama 폭으로", ""),
    # --- anchor ---
    ("layernorm_2048x768", (2048, 768), "anchor", "shape", "기존 baseline_cycle.csv", "앵커"),
    ("layernorm_512x768", (512, 768), "anchor", "exact", "기존 baseline_cycle.csv", "앵커"),
    ("layernorm_8192x768", (8192, 768), "anchor", "shape", "기존 baseline_cycle.csv", "앵커"),
    # --- added ---
    ("layernorm_2048x1024", (2048, 1024), "added", "none", "hidden 격자", ""),
    ("layernorm_2048x2048", (2048, 2048), "added", "none", "hidden 격자", ""),
    ("layernorm_2048x8192", (2048, 8192), "added", "none", "hidden 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
