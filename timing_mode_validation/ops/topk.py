"""topk — timing-mode validation cases.

size = (tokens, experts, k).  Run: `python timing_mode_validation/ops/topk.py --dtype float32`.
"""

PARAMS = "tokens, experts, k"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("topk_2048x128x8", (2048, 128, 8), "census", "none", "qwen3-moe router (실물 전문가 수)", ""),
    ("topk_2048x256x8", (2048, 256, 8), "census", "none", "deepseek-v3 router (실물)", ""),
    ("topk_2048x8x2", (2048, 8, 2), "census", "none", "qwen1.5-moe router", ""),
    # --- added ---
    ("topk_2048x16x8", (2048, 16, 8), "added", "none", "전문가 수 격자", "384는 kimi-k2 급"),
    ("topk_2048x32x8", (2048, 32, 8), "added", "none", "전문가 수 격자", "384는 kimi-k2 급"),
    ("topk_2048x384x8", (2048, 384, 8), "added", "none", "전문가 수 격자", "384는 kimi-k2 급"),
    ("topk_2048x64x8", (2048, 64, 8), "added", "none", "전문가 수 격자", "384는 kimi-k2 급"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
