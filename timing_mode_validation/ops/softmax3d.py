"""softmax3d — timing-mode validation cases.

size = (B, S_q, S_kv).  Run: `python timing_mode_validation/ops/softmax3d.py --dtype float32`.
"""

PARAMS = "B, S_q, S_kv"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("softmax3d_32x2048x2048", (32, 2048, 2048), "census", "none", "llama3-8b attention score", "실제 softmax는 [head, q, k] 3차원이다"),
    ("softmax3d_32x4096x4096", (32, 4096, 4096), "census", "none", "llama3-8b attention score", "실제 softmax는 [head, q, k] 3차원이다"),
    ("softmax3d_32x512x512", (32, 512, 512), "census", "none", "llama3-8b attention score", "실제 softmax는 [head, q, k] 3차원이다"),
    # --- added ---
    ("softmax3d_32x1024x1024", (32, 1024, 1024), "added", "none", "S 격자 보간", ""),
    ("softmax3d_32x128x128", (32, 128, 128), "added", "none", "S 격자 보간", ""),
    ("softmax3d_32x256x256", (32, 256, 256), "added", "none", "S 격자 보간", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
