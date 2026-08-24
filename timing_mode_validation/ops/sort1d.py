"""sort1d — timing-mode validation cases.

size = (N).  Run: `python timing_mode_validation/ops/sort1d.py --dtype float32`.
"""

PARAMS = "tokens, experts, k"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("sort1d_2048x128x8", (2048, 128, 8), "census", "shape", "qwen3-moe moe_infer argsort", "counting sort 재작성이 걸린 자리"),
    ("sort1d_2048x256x8", (2048, 256, 8), "census", "shape", "deepseek-v3 moe_infer argsort", ""),
    # --- added ---
    ("sort1d_512x64x6", (512, 64, 6), "added", "none", "전문가 수 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
