"""rope_outer — timing-mode validation cases.

size = (S, D).  Run: `python timing_mode_validation/ops/rope_outer.py --dtype float32`.
"""

PARAMS = "S, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("rope_outer_2048x64", (2048, 64), "census", "none", "rotary_emb inv_freq 외적", "K=1 GEMM. census 33개 모델에 있다 — 어레이 최악 모양"),
    ("rope_outer_512x64", (512, 64), "census", "none", "rotary_emb inv_freq 외적", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
