"""gemm_bias — timing-mode validation cases.

size = (M, K, N).  Run: `python timing_mode_validation/ops/gemm_bias.py --dtype float32`.
"""

PARAMS = "M, K, N"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("gemm_bias_2048x4096x4096", (2048, 4096, 4096), "census", "shape", "qwen2 q_proj (QKV bias)", "addmm 경로"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
