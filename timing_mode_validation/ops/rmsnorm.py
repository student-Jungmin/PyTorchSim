"""rmsnorm — timing-mode validation cases.

size = (rows, hidden).  Run: `python timing_mode_validation/ops/rmsnorm.py --dtype float32`.
"""

PARAMS = "rows, hidden"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("rmsnorm_128x4096", (128, 4096), "census", "shape", "llama3-8b input_layernorm", "RMSNorm은 45개 모델에 있는데 검증셋에 없다"),
    ("rmsnorm_2048x2048", (2048, 2048), "census", "shape", "gemma-2b / gemma-7b / 넓은 폭", "폭 스윕"),
    ("rmsnorm_2048x3072", (2048, 3072), "census", "shape", "gemma-2b / gemma-7b / 넓은 폭", "폭 스윕"),
    ("rmsnorm_2048x4096", (2048, 4096), "census", "shape", "llama3-8b input_layernorm", "RMSNorm은 45개 모델에 있는데 검증셋에 없다"),
    ("rmsnorm_2048x8192", (2048, 8192), "census", "none", "gemma-2b / gemma-7b / 넓은 폭", "폭 스윕"),
    ("rmsnorm_4096x4096", (4096, 4096), "census", "shape", "llama3-8b input_layernorm", "RMSNorm은 45개 모델에 있는데 검증셋에 없다"),
    ("rmsnorm_512x4096", (512, 4096), "census", "shape", "llama3-8b input_layernorm", "RMSNorm은 45개 모델에 있는데 검증셋에 없다"),
    # --- added ---
    ("rmsnorm_1024x4096", (1024, 4096), "added", "shape", "행 수 격자", "r=1은 decode 한 토큰"),
    ("rmsnorm_16384x4096", (16384, 4096), "added", "shape", "행 수 격자", "r=1은 decode 한 토큰"),
    ("rmsnorm_1x4096", (1, 4096), "added", "shape", "행 수 격자", "r=1은 decode 한 토큰"),
    ("rmsnorm_2048x1024", (2048, 1024), "added", "shape", "hidden 격자", "5120/6144/12288은 출하 폭"),
    ("rmsnorm_2048x12288", (2048, 12288), "added", "none", "hidden 격자", "5120/6144/12288은 출하 폭"),
    ("rmsnorm_2048x16384", (2048, 16384), "added", "none", "hidden 격자", "5120/6144/12288은 출하 폭"),
    ("rmsnorm_2048x512", (2048, 512), "added", "shape", "hidden 격자", "5120/6144/12288은 출하 폭"),
    ("rmsnorm_2048x5120", (2048, 5120), "added", "shape", "hidden 격자", "5120/6144/12288은 출하 폭"),
    ("rmsnorm_2048x6144", (2048, 6144), "added", "none", "hidden 격자", "5120/6144/12288은 출하 폭"),
    ("rmsnorm_64x4096", (64, 4096), "added", "shape", "행 수 격자", "r=1은 decode 한 토큰"),
    ("rmsnorm_8192x4096", (8192, 4096), "added", "shape", "행 수 격자", "r=1은 decode 한 토큰"),
    ("rmsnorm_8x4096", (8, 4096), "added", "shape", "행 수 격자", "r=1은 decode 한 토큰"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
