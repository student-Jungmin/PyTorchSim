"""attn_pv — timing-mode validation cases.

size = (B, S, D).  Run: `python timing_mode_validation/ops/attn_pv.py --dtype float32`.
"""

PARAMS = "B, S, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("attn_pv_32x128x128", (32, 128, 128), "census", "shape", "llama3-8b PV", ""),
    ("attn_pv_32x2048x128", (32, 2048, 128), "census", "shape", "llama3-8b PV", ""),
    ("attn_pv_32x4096x128", (32, 4096, 128), "census", "shape", "llama3-8b PV", ""),
    ("attn_pv_32x512x128", (32, 512, 128), "census", "shape", "llama3-8b PV", ""),
    # --- added ---
    ("attn_pv_1x16384x128", (1, 16384, 128), "added", "shape", "긴 컨텍스트", ""),
    ("attn_pv_1x8192x128", (1, 8192, 128), "added", "shape", "긴 컨텍스트", ""),
    ("attn_pv_4x8192x128", (4, 8192, 128), "added", "shape", "긴 컨텍스트", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
