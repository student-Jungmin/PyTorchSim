"""dispatch — timing-mode validation cases.

size = (tokens, k, hidden).  Run: `python timing_mode_validation/ops/dispatch.py --dtype float32`.
"""

PARAMS = "tokens, k, hidden"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("dispatch_2048x8x4096", (2048, 8, 4096), "census", "none", "MoE 토큰 분배/취합", "index_select + index_add"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
