"""moe_route — timing-mode validation cases.

size = (tokens, experts, k).  Run: `python timing_mode_validation/ops/moe_route.py --dtype float32`.
"""

PARAMS = "tokens, experts, k"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("moe_route_2048x128x8", (2048, 128, 8), "census", "none", "qwen3-moe gate 전체", "softmax+topk+renorm"),
    ("moe_route_2048x256x8", (2048, 256, 8), "census", "none", "deepseek-v3 gate 전체", ""),
    # --- added ---
    ("moe_route_2048x64x6", (2048, 64, 6), "added", "none", "전문가 수 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
