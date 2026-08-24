"""gqa_attn — timing-mode validation cases.

size = (H_q, H_kv, S, D).  Run: `python timing_mode_validation/ops/gqa_attn.py --dtype float32`.
"""

PARAMS = "H_q, H_kv, S, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("gqa_attn_16x1x2048x256", (16, 1, 2048, 256), "census", "shape", "gemma-2b MQA", "1개 kv head를 16개로"),
    ("gqa_attn_28x4x2048x128", (28, 4, 2048, 128), "census", "shape", "qwen2.5-7b n_rep=7", "홀수 복제"),
    ("gqa_attn_32x8x2048x128", (32, 8, 2048, 128), "census", "shape", "llama3-8b repeat_kv 포함", "GQA 복제 비용"),
    # --- added ---
    ("gqa_attn_32x16x2048x128", (32, 16, 2048, 128), "added", "shape", "n_rep=2", "MHA~MQA 사이"),
    ("gqa_attn_32x2x2048x128", (32, 2, 2048, 128), "added", "shape", "n_rep=16", "MHA~MQA 사이"),
    ("gqa_attn_32x32x2048x128", (32, 32, 2048, 128), "added", "shape", "n_rep=1", "MHA~MQA 사이"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
