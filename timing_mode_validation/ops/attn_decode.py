"""attn_decode — timing-mode validation cases.

size = (B, S_kv, D).  Run: `python timing_mode_validation/ops/attn_decode.py --dtype float32`.
"""

PARAMS = "B, S_kv, D"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("attn_decode_32x2048x128", (32, 2048, 128), "census", "shape", "llama3-8b decode (KV 2048)", "M=1. 서빙의 지배적 모양이고 완전히 DRAM bound"),
    ("attn_decode_32x4096x128", (32, 4096, 128), "census", "shape", "llama3-8b decode (KV 4096)", ""),
    # --- added ---
    ("attn_decode_32x1024x128", (32, 1024, 128), "added", "shape", "KV 캐시 길이 스윕", "서빙에서 실제로 움직이는 축"),
    ("attn_decode_32x16384x128", (32, 16384, 128), "added", "shape", "KV 캐시 길이 스윕", "서빙에서 실제로 움직이는 축"),
    ("attn_decode_32x32768x128", (32, 32768, 128), "added", "shape", "KV 캐시 길이 스윕", "서빙에서 실제로 움직이는 축"),
    ("attn_decode_32x512x128", (32, 512, 128), "added", "shape", "KV 캐시 길이 스윕", "서빙에서 실제로 움직이는 축"),
    ("attn_decode_32x8192x128", (32, 8192, 128), "added", "shape", "KV 캐시 길이 스윕", "서빙에서 실제로 움직이는 축"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
