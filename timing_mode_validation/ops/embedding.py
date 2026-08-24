"""embedding — timing-mode validation cases.

size = (vocab, hidden, tokens).  Run: `python timing_mode_validation/ops/embedding.py --dtype float32`.
"""

PARAMS = "vocab, hidden, tokens"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("embedding_128256x4096x512", (128256, 4096, 512), "census", "none", "llama3-8b embed_tokens (실물 vocab)", "2.1GB 테이블"),
    ("embedding_32000x4096x2048", (32000, 4096, 2048), "census", "none", "llama3-8b embed_tokens (vocab 축소)", ""),
    # --- added ---
    ("embedding_262144x4096x2048", (262144, 4096, 2048), "added", "none", "vocab 격자", "50257=gpt2, 262144=gemma3"),
    ("embedding_50257x4096x2048", (50257, 4096, 2048), "added", "none", "vocab 격자", "50257=gpt2, 262144=gemma3"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
