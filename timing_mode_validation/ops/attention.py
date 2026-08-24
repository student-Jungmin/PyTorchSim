"""attention — timing-mode validation cases.

size = (B, S, D).  Run: `python timing_mode_validation/ops/attention.py --dtype float32`.
"""

PARAMS = "B, S, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("attention_12x512x64", (12, 512, 64), "census", "exact", "기존 attention_12x512x64", "앵커"),
    ("attention_32x2048x128", (32, 2048, 128), "census", "shape", "llama3-8b attention 블록", "QK^T+softmax+PV 한 덩어리"),
    ("attention_32x512x128", (32, 512, 128), "census", "shape", "llama3-8b attention 블록", "QK^T+softmax+PV 한 덩어리"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
