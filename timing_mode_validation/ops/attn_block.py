"""attn_block — timing-mode validation cases.

size = (S, hidden, H_q, H_kv).  Run: `python timing_mode_validation/ops/attn_block.py --dtype float32`.
"""

PARAMS = "S, hidden, H_q, H_kv"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("attn_block_2048x4096x32x8", (2048, 4096, 32, 8), "census", "none", "llama3-8b attention 블록 전체", "부분의 합이 전체와 맞는지 보는 자리"),
    ("attn_block_512x4096x32x8", (512, 4096, 32, 8), "census", "none", "llama3-8b attention 블록 전체", ""),
    # --- added ---
    ("attn_block_128x4096x32x8", (128, 4096, 32, 8), "added", "none", "블록 seq 격자", ""),
    ("attn_block_4096x4096x32x8", (4096, 4096, 32, 8), "added", "none", "블록 seq 격자", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
