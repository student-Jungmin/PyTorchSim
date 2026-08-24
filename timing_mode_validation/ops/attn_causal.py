"""attn_causal — timing-mode validation cases.

size = (B, S, D).  Run: `python timing_mode_validation/ops/attn_causal.py --dtype float32`.
"""

PARAMS = "B, S, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("attn_causal_32x2048x128", (32, 2048, 128), "census", "shape", "causal mask 포함", "마스크 add의 비용"),
    ("attn_causal_32x512x128", (32, 512, 128), "census", "shape", "causal mask 포함", "마스크 add의 비용"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
