"""reduce_sum — timing-mode validation cases.

size = (rows, cols).  Run: `python timing_mode_validation/ops/reduce_sum.py --dtype float32`.
"""

PARAMS = "rows, cols"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("reduce_sum_2048x4096", (2048, 4096), "census", "shape", "norm/softmax의 리덕션 축", ""),
    # --- added ---
    ("reduce_sum_512x4096", (512, 4096), "added", "shape", "행 수 격자", ""),
    ("reduce_sum_8192x4096", (8192, 4096), "added", "shape", "행 수 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
