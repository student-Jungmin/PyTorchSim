"""softmax — timing-mode validation cases.

size = (rows, cols).  Run: `python timing_mode_validation/ops/softmax.py --dtype float32`.
"""

PARAMS = "rows, cols"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- anchor ---
    ("softmax_2048x2048", (2048, 2048), "anchor", "none", "기존 baseline_cycle.csv", "앵커"),
    ("softmax_512x512", (512, 512), "anchor", "none", "기존 baseline_cycle.csv", "앵커"),
    ("softmax_8192x8192", (8192, 8192), "anchor", "none", "기존 baseline_cycle.csv", "앵커"),
    # --- added ---
    ("softmax_1024x128256", (1024, 128256), "added", "none", "vocab softmax", "행이 아주 넓은 경우"),
    ("softmax_4096x4096", (4096, 4096), "added", "none", "2의 거듭제곱 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
