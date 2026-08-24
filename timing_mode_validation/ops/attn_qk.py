"""attn_qk — timing-mode validation cases.

size = (B(=batch*heads), S, D(head_dim)).  Run: `python timing_mode_validation/ops/attn_qk.py --dtype float32`.
"""

PARAMS = "B(=batch*heads), S, D(head_dim)"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("attn_qk_12x512x64", (12, 512, 64), "census", "exact", "bert-base / gpt2", "기존 attention_12x512x64와 같은 폭"),
    ("attn_qk_16x2048x192", (16, 2048, 192), "census", "shape", "deepseek-v3 MLA", "head_dim 192"),
    ("attn_qk_16x577x88", (16, 577, 88), "census", "exact", "llama4-vision", "head_dim 88 — 128 배수가 아니다"),
    ("attn_qk_32x128x128", (32, 128, 128), "census", "shape", "llama3-8b QK^T (B·H=32, head_dim 128)", "S가 실물. 스코어는 S^2로 큰다"),
    ("attn_qk_32x2048x128", (32, 2048, 128), "census", "shape", "llama3-8b QK^T (B·H=32, head_dim 128)", "S가 실물. 스코어는 S^2로 큰다"),
    ("attn_qk_32x4096x128", (32, 4096, 128), "census", "shape", "llama3-8b QK^T (B·H=32, head_dim 128)", "S가 실물. 스코어는 S^2로 큰다"),
    ("attn_qk_32x512x128", (32, 512, 128), "census", "shape", "llama3-8b QK^T (B·H=32, head_dim 128)", "S가 실물. 스코어는 S^2로 큰다"),
    ("attn_qk_8x2048x256", (8, 2048, 256), "census", "shape", "gemma-2b MQA", "head_dim 256, kv head 1"),
    # --- added ---
    ("attn_qk_1x16384x128", (1, 16384, 128), "added", "shape", "긴 컨텍스트", "스코어가 S^2이라 헤드 수를 줄였다"),
    ("attn_qk_1x8192x128", (1, 8192, 128), "added", "shape", "긴 컨텍스트", "스코어가 S^2이라 헤드 수를 줄였다"),
    ("attn_qk_32x1024x128", (32, 1024, 128), "added", "shape", "S 격자 보간", ""),
    ("attn_qk_32x256x128", (32, 256, 128), "added", "shape", "S 격자 보간", ""),
    ("attn_qk_32x512x160", (32, 512, 160), "added", "none", "head_dim 격자", "96/160은 캡처에 없다"),
    ("attn_qk_32x512x32", (32, 512, 32), "added", "shape", "head_dim 격자", "96/160은 캡처에 없다"),
    ("attn_qk_32x512x512", (32, 512, 512), "added", "shape", "head_dim 격자", "96/160은 캡처에 없다"),
    ("attn_qk_32x512x96", (32, 512, 96), "added", "shape", "head_dim 격자", "96/160은 캡처에 없다"),
    ("attn_qk_4x8192x128", (4, 8192, 128), "added", "shape", "긴 컨텍스트", "스코어가 S^2이라 헤드 수를 줄였다"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
