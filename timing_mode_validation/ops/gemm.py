"""gemm — timing-mode validation cases.

size = (M, K, N).  Run: `python timing_mode_validation/ops/gemm.py --dtype float32`.
"""

PARAMS = "M, K, N"

#: (name, size, origin, census, source, note).  Model-measured shapes first.
#: origin  census | sweep (one axis moved) | anchor (has a TPUv3 number) | added
#: census  exact (a captured model ran it) | shape (weights real, rows ours) | none
CASES = [
    # --- census ---
    ("gemm_128x4096x14336", (128, 4096, 14336), "census", "shape", "llama3-8b gate_proj", "SwiGLU를 seq 축으로 스윕"),
    ("gemm_16x4096x4096", (16, 4096, 4096), "census", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_197x768x768", (197, 768, 768), "census", "exact", "vit-b/16 @ 224", "M=197, 128 배수가 아니다"),
    ("gemm_2048x14336x4096", (2048, 14336, 4096), "census", "shape", "llama3-8b down_proj", "K가 깊다"),
    ("gemm_2048x192x2048", (2048, 192, 2048), "census", "none", "deepseek-v3 MLA head_dim 192", "K가 128 배수가 아니다"),
    ("gemm_2048x3072x24576", (2048, 3072, 24576), "census", "shape", "gemma-7b gate_proj", "스위트에서 가장 넓은 interm"),
    ("gemm_2048x3584x18944", (2048, 3584, 18944), "census", "shape", "qwen2.5-7b gate_proj", "실물 7B 폭"),
    ("gemm_2048x4096x1024", (2048, 4096, 1024), "census", "shape", "llama3-8b k_proj/v_proj", "GQA 8 kv head — N이 좁다"),
    ("gemm_2048x4096x1088", (2048, 4096, 1088), "census", "none", "deepseek-v2-lite kv_b_proj", "N=1088"),
    ("gemm_2048x4096x14336", (2048, 4096, 14336), "census", "shape", "llama3-8b gate_proj/up_proj", "SwiGLU 확장"),
    ("gemm_2048x4096x32000", (2048, 4096, 32000), "census", "none", "llama3-8b lm_head (vocab 축소)", "N이 크다 — weight 스트리밍"),
    ("gemm_2048x5120x8192", (2048, 5120, 8192), "census", "shape", "llama4-scout expert", "MoE 전문가 하나"),
    ("gemm_256x5120x8192", (256, 5120, 8192), "census", "shape", "llama4-scout expert (top-1 분배 후)", "전문가당 토큰 수가 M"),
    ("gemm_32x4096x4096", (32, 4096, 4096), "census", "exact", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_4096x4096x14336", (4096, 4096, 14336), "census", "shape", "llama3-8b gate_proj", "SwiGLU를 seq 축으로 스윕"),
    ("gemm_512x3072x768", (512, 3072, 768), "census", "exact", "bert-base output.dense", ""),
    ("gemm_512x4096x128256", (512, 4096, 128256), "census", "none", "llama3-8b lm_head (실물 vocab) / N 스윕", "2.1GB 가중치, DRAM bound. heavy"),
    ("gemm_512x4096x14336", (512, 4096, 14336), "census", "shape", "llama3-8b gate_proj / N 스윕", "SwiGLU를 seq 축으로 스윕"),
    ("gemm_512x768x3072", (512, 768, 3072), "census", "exact", "bert-base intermediate.dense", ""),
    ("gemm_512x768x768", (512, 768, 768), "census", "exact", "bert-base q/k/v/o @ seq 512", "bert_base 앵커와 같은 폭"),
    ("gemm_577x1408x1408", (577, 1408, 1408), "census", "exact", "llama4-vision q_proj", "M도 K도 128 배수가 아니다"),
    ("gemm_577x1408x5632", (577, 1408, 5632), "census", "exact", "llama4-vision mlp.fc1", "M=577"),
    # --- sweep ---
    ("gemm_1024x4096x4096", (1024, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_128x4096x4096", (128, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_1x4096x4096", (1, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_2048x4096x4096", (2048, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_256x4096x4096", (256, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_4096x4096x4096", (4096, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj / 2의 거듭제곱 격자", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_512x128x4096", (512, 128, 4096), "sweep", "none", "K 스윕", "누적 깊이만 바꾼다"),
    ("gemm_512x14336x4096", (512, 14336, 4096), "sweep", "shape", "K 스윕", "누적 깊이만 바꾼다"),
    ("gemm_512x2048x4096", (512, 2048, 4096), "sweep", "shape", "K 스윕", "누적 깊이만 바꾼다"),
    ("gemm_512x24576x4096", (512, 24576, 4096), "sweep", "none", "K 스윕", "누적 깊이만 바꾼다"),
    ("gemm_512x4096x1024", (512, 4096, 1024), "sweep", "shape", "N 스윕", "가중치 스트리밍만 바꾼다"),
    ("gemm_512x4096x128", (512, 4096, 128), "sweep", "none", "N 스윕", "가중치 스트리밍만 바꾼다"),
    ("gemm_512x4096x32768", (512, 4096, 32768), "sweep", "none", "N 스윕", "가중치 스트리밍만 바꾼다"),
    ("gemm_512x4096x4096", (512, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj / K 스윕 / N 스윕", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    ("gemm_512x512x4096", (512, 512, 4096), "sweep", "none", "K 스윕", "누적 깊이만 바꾼다"),
    ("gemm_64x4096x4096", (64, 4096, 4096), "sweep", "shape", "llama3-8b q_proj/o_proj", "M=seq. 128x128 어레이 높이를 건너는 스윕 — census GEMM의 85%가 M<128"),
    # --- anchor ---
    ("gemm_1024x1024x1024", (1024, 1024, 1024), "anchor", "shape", "기존 baseline_cycle.csv", "이미 TPUv3 수치가 있는 앵커"),
    ("gemm_2048x2048x2048", (2048, 2048, 2048), "anchor", "shape", "기존 baseline_cycle.csv", "이미 TPUv3 수치가 있는 앵커"),
    ("gemm_256x256x256", (256, 256, 256), "anchor", "shape", "기존 baseline_cycle.csv", "이미 TPUv3 수치가 있는 앵커"),
    ("gemm_512x512x512", (512, 512, 512), "anchor", "shape", "기존 baseline_cycle.csv / K=N 격자", "이미 TPUv3 수치가 있는 앵커"),
    # --- added ---
    ("gemm_128x128x128", (128, 128, 128), "added", "shape", "2의 거듭제곱 격자", "정사각 곡선의 양 끝"),
    ("gemm_128x16384x16384", (128, 16384, 16384), "added", "none", "short-fat", "M만 작다"),
    ("gemm_1536x4096x4096", (1536, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_192x4096x4096", (192, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_1x8192x8192", (1, 8192, 8192), "added", "none", "GEMV", "decode의 행렬-벡터 극단"),
    ("gemm_2048x12288x12288", (2048, 12288, 12288), "added", "none", "GPT-3 175B q_proj (캡처 안 된 출하 폭)", ""),
    ("gemm_2048x4096x11008", (2048, 4096, 11008), "added", "none", "llama-7b gate_proj (캡처 안 된 출하 폭)", ""),
    ("gemm_2048x5120x13824", (2048, 5120, 13824), "added", "none", "llama-13b gate_proj (캡처 안 된 출하 폭)", ""),
    ("gemm_2048x5120x5120", (2048, 5120, 5120), "added", "shape", "llama-13b q_proj (캡처 안 된 출하 폭)", ""),
    ("gemm_2048x6656x17920", (2048, 6656, 17920), "added", "none", "llama-33b gate_proj (캡처 안 된 출하 폭)", ""),
    ("gemm_2048x8192x28672", (2048, 8192, 28672), "added", "none", "llama-70b gate_proj (캡처 안 된 출하 폭)", ""),
    ("gemm_3072x4096x4096", (3072, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_384x4096x4096", (384, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_512x1024x1024", (512, 1024, 1024), "added", "shape", "K=N 격자", "M 고정, 문제 크기만"),
    ("gemm_512x1024x4096", (512, 1024, 4096), "added", "shape", "K 격자 확장", ""),
    ("gemm_512x16384x16384", (512, 16384, 16384), "added", "none", "K=N 격자", "M 고정, 문제 크기만"),
    ("gemm_512x16384x4096", (512, 16384, 4096), "added", "none", "K 격자 확장", ""),
    ("gemm_512x2048x2048", (512, 2048, 2048), "added", "shape", "K=N 격자", "M 고정, 문제 크기만"),
    ("gemm_512x256x4096", (512, 256, 4096), "added", "none", "K 격자 확장", ""),
    ("gemm_512x32768x4096", (512, 32768, 4096), "added", "none", "K 격자 확장", ""),
    ("gemm_512x4096x16384", (512, 4096, 16384), "added", "none", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x4096x2048", (512, 4096, 2048), "added", "shape", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x4096x256", (512, 4096, 256), "added", "none", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x4096x262144", (512, 4096, 262144), "added", "none", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x4096x512", (512, 4096, 512), "added", "none", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x4096x65536", (512, 4096, 65536), "added", "none", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x4096x8192", (512, 4096, 8192), "added", "shape", "N 격자 확장", "262144는 gemma3 급 vocab"),
    ("gemm_512x64x4096", (512, 64, 4096), "added", "shape", "K 격자 확장", ""),
    ("gemm_512x8192x4096", (512, 8192, 4096), "added", "none", "K 격자 확장", ""),
    ("gemm_512x8192x8192", (512, 8192, 8192), "added", "none", "K=N 격자", "M 고정, 문제 크기만"),
    ("gemm_768x4096x4096", (768, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_8192x128x4096", (8192, 128, 4096), "added", "none", "tall-shallow", "K가 얕다"),
    ("gemm_8192x4096x128", (8192, 4096, 128), "added", "none", "tall-skinny", "N이 어레이 한 폭"),
    ("gemm_8192x4096x4096", (8192, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_8192x8192x8192", (8192, 8192, 8192), "added", "none", "2의 거듭제곱 격자", "정사각 곡선의 양 끝"),
    ("gemm_8x4096x4096", (8, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
    ("gemm_96x4096x4096", (96, 4096, 4096), "added", "shape", "M 스윕 보간", "128/256 부근이 v6e에서 꺾이는 곳이라 촘촘히 본다"),
]

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run import main

    main(__file__)
