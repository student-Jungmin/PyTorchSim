"""pwconv — timing-mode validation cases.

size = (N, C_in, H, W, C_out).  Run: `python timing_mode_validation/ops/pwconv.py --dtype float32`.
"""

PARAMS = "N, C_in, H, W, C_out"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("pwconv_1x256x56x56x128", (1, 256, 56, 56, 128), "census", "exact", "yolo/mobilenet 1x1", "census 최다 conv 종류"),
    ("pwconv_1x512x28x28x256", (1, 512, 28, 28, 256), "census", "exact", "yolo/mobilenet 1x1", ""),
    # --- added ---
    ("pwconv_1x1024x14x14x256", (1, 1024, 14, 14, 256), "added", "exact", "resnet50 1x1", ""),
    ("pwconv_1x2048x7x7x512", (1, 2048, 7, 7, 512), "added", "exact", "resnet50 1x1", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
