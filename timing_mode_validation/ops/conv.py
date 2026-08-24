"""conv — timing-mode validation cases.

size = (N, H, W, C_in, C_out, R, stride, pad).  Run: `python timing_mode_validation/ops/conv.py --dtype float32`.
"""

PARAMS = "N, H, W, C_in, C_out, R, stride, pad"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("conv_1x14x14x256x256x3x1x1", (1, 14, 14, 256, 256, 3, 1, 1), "census", "exact", "resnet, batch 1", "batch 축의 효과"),
    ("conv_1x28x28x128x128x3x1x1", (1, 28, 28, 128, 128, 3, 1, 1), "census", "exact", "resnet, batch 1", "batch 축의 효과"),
    ("conv_1x320x320x64x64x3x1x1", (1, 320, 320, 64, 64, 3, 1, 1), "census", "shape", "yolo @ 실물 640 입력", "테스트는 64x64로 도는데 실물은 640이다"),
    ("conv_1x56x56x64x64x3x1x1", (1, 56, 56, 64, 64, 3, 1, 1), "census", "exact", "resnet, batch 1", "batch 축의 효과"),
    ("conv_1x7x7x512x512x3x1x1", (1, 7, 7, 512, 512, 3, 1, 1), "census", "exact", "resnet, batch 1", "batch 축의 효과"),
    # --- anchor ---
    ("conv_64x14x14x256x256x3x1x1", (64, 14, 14, 256, 256, 3, 1, 1), "anchor", "shape", "기존 baseline_cycle.csv", "앵커"),
    ("conv_64x28x28x128x128x3x1x1", (64, 28, 28, 128, 128, 3, 1, 1), "anchor", "shape", "기존 baseline_cycle.csv", "앵커"),
    ("conv_64x56x56x64x64x3x1x1", (64, 56, 56, 64, 64, 3, 1, 1), "anchor", "shape", "기존 baseline_cycle.csv", "앵커"),
    ("conv_64x7x7x512x512x3x1x1", (64, 7, 7, 512, 512, 3, 1, 1), "anchor", "shape", "기존 baseline_cycle.csv", "앵커"),
    # --- added ---
    ("conv_16x14x14x256x256x3x1x1", (16, 14, 14, 256, 256, 3, 1, 1), "added", "shape", "batch 16", ""),
    ("conv_16x56x56x64x64x3x1x1", (16, 56, 56, 64, 64, 3, 1, 1), "added", "shape", "batch 16", "batch 축"),
    ("conv_1x224x224x3x64x7x2x3", (1, 224, 224, 3, 64, 7, 2, 3), "added", "exact", "resnet stem 7x7 s2", ""),
    ("conv_1x56x56x64x64x3x2x1", (1, 56, 56, 64, 64, 3, 2, 1), "added", "exact", "stride 2 3x3", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
