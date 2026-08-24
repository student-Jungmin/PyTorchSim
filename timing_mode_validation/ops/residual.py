"""residual — timing-mode validation cases.

size = (rows, hidden).  Run: `python timing_mode_validation/ops/residual.py --dtype float32`.
"""

PARAMS = "rows, hidden"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("residual_2048x4096", (2048, 4096), "census", "shape", "layer residual add", "가장 싼 DRAM bound 모양"),
    # --- added ---
    ("residual_1024x1024", (1024, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_16384x1024", (16384, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_16x1024", (16, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_1x1024", (1, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_256x1024", (256, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_4096x1024", (4096, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_4x1024", (4, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
    ("residual_64x1024", (64, 1024), "added", "shape", "대역폭 곡선", "1K~16M 원소. DRAM 모델을 맞추는 첫 단계"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
