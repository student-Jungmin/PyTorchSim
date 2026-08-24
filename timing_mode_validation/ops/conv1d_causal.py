"""conv1d_causal — timing-mode validation cases.

size = (N, C, L, R).  Run: `python timing_mode_validation/ops/conv1d_causal.py --dtype float32`.
"""

PARAMS = "N, C, L, R"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("conv1d_causal_1x8192x2048x4", (1, 8192, 2048, 4), "census", "shape", "qwen3-next causal Conv1d @ 실물 seq", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
