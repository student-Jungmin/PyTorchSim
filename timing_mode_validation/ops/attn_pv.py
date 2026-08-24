"""attn_pv — timing-mode validation cases.

size = (B, S, D).  Run: `python timing_mode_validation/ops/attn_pv.py --dtype float32`.
"""

PARAMS = "B, S, D"

#: ONE CASE ON PURPOSE, as a control rather than a sweep. PV has the same MAC
#: count and the same fp32 traffic as attn_qk at every (B, S, D) measured --
#: 17,179,869,184 and 603,979,776 at 32x2048x128, to the digit -- because the
#: S x S score matrix PV reads is the one attn_qk wrote. Sweeping both measures
#: one shape twice. What is NOT shared is the transpose (QK^T needs K
#: transposed, PV does not) and the read/write split, so one point stays: if it
#: comes out different from attn_qk, the difference IS one of those two.
CASES = [
    # --- census ---
    ("attn_pv_32x2048x128", (32, 2048, 128), "census", "exact", "llama3-8b PV",
     "attn_qk와 같은 MAC·같은 트래픽. 다르게 나오면 그 차이가 transpose 아니면 읽기/쓰기 비대칭"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
