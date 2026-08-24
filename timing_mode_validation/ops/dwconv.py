"""dwconv — timing-mode validation cases.

size = (N, C, H, W, R, stride).  Run: `python timing_mode_validation/ops/dwconv.py --dtype float32`.
"""

PARAMS = "N, C, H, W, R, stride"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("dwconv_1x384x14x14x3x1", (1, 384, 14, 14, 3, 1), "census", "exact", "mobilenet-v2 depthwise", ""),
    ("dwconv_1x96x112x112x3x2", (1, 96, 112, 112, 3, 2), "census", "exact", "mobilenet-v2 depthwise", "groups=C"),
    # --- added ---
    ("dwconv_1x144x56x56x3x2", (1, 144, 56, 56, 3, 2), "added", "exact", "mobilenet 격자", ""),
    ("dwconv_1x32x112x112x3x1", (1, 32, 112, 112, 3, 1), "added", "exact", "mobilenet 격자", ""),
    ("dwconv_1x576x14x14x3x1", (1, 576, 14, 14, 3, 1), "added", "exact", "mobilenet 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
