"""maxpool — timing-mode validation cases.

size = (N, C, H, W, k, stride).  Run: `python timing_mode_validation/ops/maxpool.py --dtype float32`.
"""

PARAMS = "N, C, H, W, k, stride"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("maxpool_1x512x20x20x5x1", (1, 512, 20, 20, 5, 1), "census", "none", "yolo SPPF", ""),
    ("maxpool_1x64x112x112x3x2", (1, 64, 112, 112, 3, 2), "census", "none", "resnet stem", ""),
    # --- added ---
    ("maxpool_1x64x224x224x3x2", (1, 64, 224, 224, 3, 2), "added", "none", "resnet stem 실물 해상도", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
