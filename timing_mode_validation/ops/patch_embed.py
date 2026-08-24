"""patch_embed — timing-mode validation cases.

size = (N, C, H, C_out, patch).  Run: `python timing_mode_validation/ops/patch_embed.py --dtype float32`.
"""

PARAMS = "N, C, H, C_out, patch"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("patch_embed_1x3x224x768x16", (1, 3, 224, 768, 16), "census", "exact", "vit-b/16 patch embedding", "kernel=stride"),
    ("patch_embed_1x3x336x1408x14", (1, 3, 336, 1408, 14), "census", "none", "llama4-vision patch embedding", ""),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
