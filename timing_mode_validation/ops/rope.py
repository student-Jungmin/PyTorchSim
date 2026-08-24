"""rope — timing-mode validation cases.

size = (B, H, S, D).  Run: `python timing_mode_validation/ops/rope.py --dtype float32`.
"""

PARAMS = "B, H, S, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("rope_1x32x2048x128", (1, 32, 2048, 128), "census", "none", "llama3-8b apply_rotary_pos_emb", "rope 전체"),
    ("rope_1x32x512x128", (1, 32, 512, 128), "census", "none", "llama3-8b apply_rotary_pos_emb", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
