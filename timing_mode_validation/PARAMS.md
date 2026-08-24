# 연산별 파라미터 범위

**32개 연산 / 237개 케이스.** `ops/*.py`에서 생성한다 (`python timing_mode_validation/index.py`) — 케이스를 고치면 이 파일도 다시 만들어진다.

- **범위**는 그 파라미터가 도는 최소~최대다. 축이 여러 개인 연산은 모든 조합을 도는 것이 아니라 케이스가 정한 조합만 돈다 — 조합은 `timing_cases.csv`에 있다.
- **모델 값**은 캡처된 모델(`OP_CENSUS.md`)에서 온 shape가 쓰는 값, **추가 값**은 모델에 없지만 자주 쓰이는 구간이다.
- 폭은 fp32와 fp16 두 번 돈다. bf16은 백엔드가 거부한다(README 참조).

## `attention` — 3개 (모델 3 / 추가 0)

Score, softmax and context together -- the same math as attention.py.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 12 – 32 | 12, 32 | — |
| `S` | 512 – 2048 | 512, 2048 | — |
| `D` | 64 – 128 | 64, 128 | — |

## `attn_block` — 4개 (모델 2 / 추가 2)

One attention block end to end: projections, GQA repeat, scores, context.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `S` | 128 – 4096 | 512, 2048 | 128, 4096 |
| `hidden` | 4096 | 4096 | 4096 |
| `H_q` | 32 | 32 | 32 |
| `H_kv` | 8 | 8 | 8 |

## `attn_causal` — 2개 (모델 2 / 추가 0)

Attention with the additive causal mask the models actually build.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 32 | 32 | — |
| `S` | 512 – 2048 | 512, 2048 | — |
| `D` | 128 | 128 | — |

## `attn_decode` — 7개 (모델 2 / 추가 5)

One query against a KV cache: the matrix-vector regime of serving.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 32 | 32 | 32 |
| `S_kv` | 512 – 32768 | 2048, 4096 | 512, 1024, 8192, 16384, 32768 |
| `D` | 128 | 128 | 128 |

## `attn_pv` — 1개 (모델 1 / 추가 0)

The context matmul alone: [B,S,S] @ [B,S,D].

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 32 | 32 | — |
| `S` | 2048 | 2048 | — |
| `D` | 128 | 128 | — |

## `attn_qk` — 17개 (모델 8 / 추가 9)

The score matmul alone: [B,S,D] @ [B,D,S].

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 1 – 32 | 8, 12, 16, 32 | 1, 4, 32 |
| `S` | 128 – 16384 | 128, 512, 577, 2048, 4096 | 256, 512, 1024, 8192, 16384 |
| `D` | 32 – 512 | 64, 88, 128, 192, 256 | 32, 96, 128, 160, 512 |

## `conv` — 13개 (모델 9 / 추가 4)

A dense 2-D convolution, the ResNet-shaped baseline.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 – 64 | 1, 64 | 1, 16 |
| `H` | 7 – 320 | 7, 14, 28, 56, 320 | 14, 56, 224 |
| `W` | 7 – 320 | 7, 14, 28, 56, 320 | 14, 56, 224 |
| `C_in` | 3 – 512 | 64, 128, 256, 512 | 3, 64, 256 |
| `C_out` | 64 – 512 | 64, 128, 256, 512 | 64, 256 |
| `R` | 3 – 7 | 3 | 3, 7 |
| `stride` | 1 – 2 | 1 | 1, 2 |
| `pad` | 1 – 3 | 1 | 1, 3 |

## `conv1d_causal` — 1개 (모델 1 / 추가 0)

The depthwise causal Conv1d a linear-attention block puts in the loop.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 | 1 | — |
| `C` | 8192 | 8192 | — |
| `L` | 2048 | 2048 | — |
| `R` | 4 | 4 | — |

## `convtranspose` — 1개 (모델 1 / 추가 0)

A transposed convolution, which the frontend rewrites as a direct one.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 | 1 | — |
| `C_in` | 256 | 256 | — |
| `H` | 20 | 20 | — |
| `W` | 20 | 20 | — |
| `C_out` | 128 | 128 | — |
| `R` | 2 | 2 | — |
| `stride` | 2 | 2 | — |

## `dispatch` — 1개 (모델 1 / 추가 0)

Gather tokens to experts and scatter the results back.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `tokens` | 2048 | 2048 | — |
| `k` | 8 | 8 | — |
| `hidden` | 4096 | 4096 | — |

## `dwconv` — 5개 (모델 2 / 추가 3)

Depthwise convolution: groups == channels, one filter per channel.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 | 1 | 1 |
| `C` | 32 – 576 | 96, 384 | 32, 144, 576 |
| `H` | 14 – 112 | 14, 112 | 14, 56, 112 |
| `W` | 14 – 112 | 14, 112 | 14, 56, 112 |
| `R` | 3 | 3 | 3 |
| `stride` | 1 – 2 | 1, 2 | 1, 2 |

## `embedding` — 4개 (모델 2 / 추가 2)

Token embedding: T rows gathered out of a [V, H] table.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `vocab` | 32000 – 262144 | 32000, 128256 | 50257, 262144 |
| `hidden` | 4096 | 4096 | 4096 |
| `tokens` | 512 – 2048 | 512, 2048 | 2048 |

## `gelu` — 5개 (모델 2 / 추가 3)

GELU on an MLP-width tensor.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 512 – 2048 | 512, 2048 | 2048 |
| `interm` | 3072 – 12288 | 3072 | 4096, 8192, 12288 |

## `gemm` — 79개 (모델 42 / 추가 37)

A @ B, the projection shape every transformer layer is made of.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `M` | 1 – 8192 | 1, 16, 32, 64, 128, 197, 256, 512, 577, 1024, 2048, 4096 | 1, 8, 96, 128, 192, 384, 512, 768, 1536, 2048, 3072, 8192 |
| `K` | 64 – 32768 | 128, 192, 256, 512, 768, 1024, 1408, 2048, 3072, 3584, 4096, 5120, 14336, 24576 | 64, 128, 256, 1024, 2048, 4096, 5120, 6656, 8192, 12288, 16384, 32768 |
| `N` | 128 – 262144 | 128, 256, 512, 768, 1024, 1088, 1408, 2048, 3072, 4096, 5632, … , 32000, 32768, 128256 | 128, 256, 512, 1024, 2048, 4096, 5120, 8192, 11008, 12288, 13824, … , 28672, 65536, 262144 |

## `gemm_bias` — 1개 (모델 1 / 추가 0)

A @ B + bias, the shape a projection with bias takes.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `M` | 2048 | 2048 | — |
| `K` | 4096 | 4096 | — |
| `N` | 4096 | 4096 | — |

## `gqa_attn` — 6개 (모델 3 / 추가 3)

Attention with repeat_kv in the graph, which is what GQA costs.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `H_q` | 16 – 32 | 16, 28, 32 | 32 |
| `H_kv` | 1 – 32 | 1, 4, 8 | 2, 16, 32 |
| `S` | 2048 | 2048 | 2048 |
| `D` | 128 – 256 | 128, 256 | 128 |

## `layernorm` — 7개 (모델 4 / 추가 3)

LayerNorm over the last axis, the encoder-side norm.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 512 – 8192 | 512, 2048, 8192 | 2048 |
| `hidden` | 768 – 8192 | 768, 4096 | 1024, 2048, 8192 |

## `maxpool` — 3개 (모델 2 / 추가 1)

Max pooling, the YOLO SPPF and ResNet stem shape.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 | 1 | 1 |
| `C` | 64 – 512 | 64, 512 | 64 |
| `H` | 20 – 224 | 20, 112 | 224 |
| `W` | 20 – 224 | 20, 112 | 224 |
| `k` | 3 – 5 | 3, 5 | 3 |
| `stride` | 1 – 2 | 1, 2 | 2 |

## `mlp_swiglu` — 4개 (모델 2 / 추가 2)

A whole gated MLP: three GEMMs and the elementwise between them.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `S` | 128 – 4096 | 512, 2048 | 128, 4096 |
| `hidden` | 4096 | 4096 | 4096 |
| `interm` | 14336 | 14336 | 14336 |

## `moe_route` — 3개 (모델 2 / 추가 1)

Router softmax, top-k and renormalise -- the whole gate.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `tokens` | 2048 | 2048 | 2048 |
| `experts` | 64 – 256 | 128, 256 | 64 |
| `k` | 6 – 8 | 8 | 6 |

## `patch_embed` — 2개 (모델 2 / 추가 0)

The vision patch embedding: kernel == stride, no overlap.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 | 1 | — |
| `C` | 3 | 3 | — |
| `H` | 224 – 336 | 224, 336 | — |
| `C_out` | 768 – 1408 | 768, 1408 | — |
| `patch` | 14 – 16 | 14, 16 | — |

## `pwconv` — 4개 (모델 2 / 추가 2)

Pointwise 1x1 convolution, the most common conv node in the census.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `N` | 1 | 1 | 1 |
| `C_in` | 256 – 2048 | 256, 512 | 1024, 2048 |
| `H` | 7 – 56 | 28, 56 | 7, 14 |
| `W` | 7 – 56 | 28, 56 | 7, 14 |
| `C_out` | 128 – 512 | 128, 256 | 256, 512 |

## `reduce_sum` — 3개 (모델 1 / 추가 2)

Row-wise sum, the reduction shape a norm and a softmax share.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 512 – 8192 | 2048 | 512, 8192 |
| `cols` | 4096 | 4096 | 4096 |

## `residual` — 9개 (모델 1 / 추가 8)

A plain residual add, the cheapest DRAM-bound shape in a layer.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 1 – 16384 | 2048 | 1, 4, 16, 64, 256, 1024, 4096, 16384 |
| `hidden` | 1024 – 4096 | 4096 | 1024 |

## `rmsnorm` — 19개 (모델 7 / 추가 12)

RMSNorm over the last axis -- 45 of the 68 captured models use it.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 1 – 16384 | 128, 512, 2048, 4096 | 1, 8, 64, 1024, 2048, 8192, 16384 |
| `hidden` | 512 – 16384 | 2048, 3072, 4096, 8192 | 512, 1024, 4096, 5120, 6144, 12288, 16384 |

## `rope` — 2개 (모델 2 / 추가 0)

Rotary embedding applied to one projection, rotate_half form.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 1 | 1 | — |
| `H` | 32 | 32 | — |
| `S` | 512 – 2048 | 512, 2048 | — |
| `D` | 128 | 128 | — |

## `rope_outer` — 2개 (모델 2 / 추가 0)

The inv_freq outer product: a matmul whose K is 1.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `S` | 512 – 2048 | 512, 2048 | — |
| `D` | 64 | 64 | — |

## `softmax` — 5개 (모델 3 / 추가 2)

Row softmax, the shape attention scores arrive in.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 512 – 8192 | 512, 2048, 8192 | 1024, 4096 |
| `cols` | 512 – 128256 | 512, 2048, 8192 | 4096, 128256 |

## `softmax3d` — 6개 (모델 3 / 추가 3)

Softmax on a [head, query, key] score tensor, without the matmuls.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `B` | 32 | 32 | 32 |
| `S_q` | 128 – 4096 | 512, 2048, 4096 | 128, 256, 1024 |
| `S_kv` | 128 – 4096 | 512, 2048, 4096 | 128, 256, 1024 |

## `sort1d` — 3개 (모델 2 / 추가 1)

The router's argsort of flattened top-k ids -- what counting sort replaces.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `tokens` | 512 – 2048 | 2048 | 512 |
| `experts` | 64 – 256 | 128, 256 | 64 |
| `k` | 6 – 8 | 8 | 6 |

## `swiglu` — 8개 (모델 3 / 추가 5)

silu(gate) * up, the elementwise half of a gated MLP.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `rows` | 512 – 2048 | 512, 2048 | 2048 |
| `interm` | 4096 – 28672 | 14336, 24576 | 4096, 8192, 11008, 16384, 28672 |

## `topk` — 7개 (모델 3 / 추가 4)

Top-k over per-token expert logits, the MoE router's first step.

| 파라미터 | 범위 | 모델 값 | 추가 값 |
|---|---|---|---|
| `tokens` | 2048 | 2048 | 2048 |
| `experts` | 8 – 384 | 8, 128, 256 | 16, 32, 64, 384 |
| `k` | 2 – 8 | 2, 8 | 8 |
