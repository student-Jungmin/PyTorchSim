"""convtranspose — timing-mode validation cases.

size = (N, C_in, H, W, C_out, R, stride).  Run: `python timing_mode_validation/ops/convtranspose.py --dtype float32`.
"""

PARAMS = "N, C_in, H, W, C_out, R, stride"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("convtranspose_1x256x20x20x128x2x2", (1, 256, 20, 20, 128, 2, 2), "census", "none", "yolov6 neck", "직접 conv로 재작성됨"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
