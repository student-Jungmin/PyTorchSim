"""Llama 4 on the Triton codegen route.

Built from ``Llama4TextConfig`` with random weights, so it needs no network and
no checkpoint. Needs transformers 4.51.3 -- the module does not exist at 4.43.

WHAT IS NEW, and it is four things rather than one. Llama 4 is the first break
in the family since 3, and every one of these is absent from test_llama3x:

  MoE in EVERY layer    interleave_moe_layer_step defaults to 1, so unlike
                        Mixtral or Qwen-MoE there is no dense layer to fall back
                        on. Scout routes top-1 of 16 experts and adds a SHARED
                        expert whose output is summed in unconditionally.

  NoPE layers           no_rope_layers is [1,1,1,0,...] -- every fourth layer
                        applies no rotary embedding at all. A decoder where the
                        position information is present in some layers and
                        absent in others has no precedent in this suite, and it
                        is a graph difference rather than a constant one.

  QK norm               use_qk_norm puts an RMSNorm on q and k after the rotary,
                        per head rather than per hidden.

  chunked attention     attention_chunk_size bounds how far back a token may
                        attend, which is a second mask shape beside the causal
                        one.

The presets keep all four. Experts are scaled DOWN -- 4 rather than Scout's 16 --
on the same argument tests/models/Qwen/test_qwen.py makes for its MoE presets:
what is unproven is the routing and the combine, and those do not change with
the expert count. Layers stay at 4 because that is the shortest run that
contains a NoPE layer; `scout` names its own no_rope_layers to get one in two.

    source /workspace/tnpu-env.sh
    python tests/models/Llama/test_llama4.py --preset small

STATUS: bring-up. Not in scripts/ci/triton_route_passing.txt until it passes.

MEASURED 2026-08-13 on transformers 5.15.0, tnpu 983eee4, --preset small:
147,874,816 parameters, all four guards green -- 4/4 MoE layers,
no_rope_layers [1,1,1,0], experts=4 top-1, qk_norm, chunk=32 -- and then, at 61s:

    AssertionError: expected size 32==32, stride 64==1 at dim=1;
                    expected size 64==64, stride 1==32 at dim=2
    Error in op: torch.ops.aten.polar.default

TWO WALLS, IN ORDER. `polar` is the complex-valued rotary: Llama 4 builds its
frequencies as complex numbers (`polar`, `view_as_complex`, `view_as_real`),
which no other model in this suite does, and the strides it arrives with are
transposed against what the op expects. Behind that sits the MoE combine's
`tl.atomic_add`, which is where this stopped on transformers 4.51.3:

    kernel 28 ..._scatter_add_view_28
    error: unexpected op in ptr sequence

That one is triton_shared's PtrAnalysis meeting `tt.atomic_rmw`, an op it has
no arm for -- no coverage kernel in triton-npu calls `tl.atomic_*` at all, and
the two kernels that look like they do (dl/moe_combine, dl/embedding_grad_
scatter_add) both AVOID atomics on purpose, one with a read-modify-write and
one by resolving the collision as a predicate.

The atomic is substitutable rather than needed. HF's own comment says why: "we
have to do this because we used all experts on all tokens". The index is
`arange(T)` tiled over the expert axis -- NOT the top-k indices, which are
overwritten a few lines above -- so the scatter_add is a sum over the expert
axis with a statically known 4-way collision and no data dependence. Inductor
already folded it that far: the emitted address is `x0 + 1024*(x1 % 32)`,
computed from the iteration index rather than loaded.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# Experts scaled down, everything that decides the SHAPE of the routing kept.
# `layers` is 4 so no_rope_layers' period lands a NoPE layer inside the run.
_PRESETS = {
    "tiny": dict(layers=4, hidden=256, heads=8, kv_heads=2, intermediate=512,
                 experts=4, vocab=256, seq=16, chunk=32),

    "small": dict(layers=4, hidden=1024, heads=16, kv_heads=4, intermediate=2048,
                  experts=4, vocab=1024, seq=32, chunk=32),

    # Scout's real width and head ratio, two layers so one of them is NoPE, and
    # four experts rather than sixteen.
    "scout": dict(layers=2, hidden=5120, heads=40, kv_heads=8, intermediate=8192,
                  experts=4, vocab=2048, seq=32, chunk=32,
                  no_rope=[1, 0]),
}


def _dtype_from_str(name):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float32)


def _build_config(preset, seq_len):
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

    p = _PRESETS[preset]
    seq_len = seq_len if seq_len is not None else p["seq"]
    kw = dict(
        vocab_size=p["vocab"], hidden_size=p["hidden"],
        num_attention_heads=p["heads"], num_key_value_heads=p["kv_heads"],
        intermediate_size=p["intermediate"], intermediate_size_mlp=p["intermediate"],
        num_hidden_layers=p["layers"],
        num_local_experts=p["experts"], num_experts_per_tok=1,
        interleave_moe_layer_step=1,
        max_position_embeddings=max(seq_len, 512), rope_theta=500000.0,
        use_qk_norm=True, attention_chunk_size=p["chunk"],
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_cache=False, attn_implementation="eager",
    )
    if "no_rope" in p:
        kw["no_rope_layers"] = p["no_rope"]
    return Llama4TextConfig(**kw), seq_len


def _assert_character(cfg, model):
    """Fail if the model did not get the four things this file is about."""
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    kinds = [type(l.feed_forward).__name__ for l in model.layers]
    n_moe = sum(isinstance(l.feed_forward, Llama4TextMoe) for l in model.layers)
    assert n_moe == len(model.layers), \
        f"every Llama 4 layer is MoE; got {n_moe} of {len(model.layers)}: {kinds}"

    # A run with no NoPE layer is a Llama 3 decoder with extra steps.
    assert 0 in list(cfg.no_rope_layers), \
        f"preset must contain a NoPE layer; no_rope_layers={list(cfg.no_rope_layers)}"

    assert cfg.use_qk_norm, "preset must keep the q/k norm"
    assert cfg.num_experts_per_tok < cfg.num_local_experts, \
        "top-k must be smaller than the expert count or the routing is dense"
    print(f"guards ok: {n_moe}/{len(model.layers)} MoE layers, "
          f"no_rope_layers={list(cfg.no_rope_layers)}, "
          f"experts={cfg.num_local_experts} top-{cfg.num_experts_per_tok}, "
          f"qk_norm={cfg.use_qk_norm}, chunk={cfg.attention_chunk_size}")


@torch.no_grad()
def run_llama4(device, preset="small", batch=1, seq_len=None, dtype="float32",
               compile_model=True, rtol=1e-2, atol=1e-2):
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel

    torch_dtype = _dtype_from_str(dtype)
    cfg, seq_len = _build_config(preset, seq_len)

    torch.manual_seed(0)
    model_cpu = Llama4TextModel(cfg).to(dtype=torch_dtype).eval()

    print(f"preset={preset} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"interm={cfg.intermediate_size} vocab={cfg.vocab_size} seq={seq_len}")
    print("model params:", sum(p.numel() for p in model_cpu.parameters()))
    _assert_character(cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(3, cfg.vocab_size, (batch, seq_len), generator=g,
                              dtype=torch.long)

    cpu_out = model_cpu(input_ids=input_ids).last_hidden_state

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = model_npu(input_ids=input_ids.to(device)).last_hidden_state

    test_result(f"Llama 4 text ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Llama 4 Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 4 on the Triton route")
    parser.add_argument("--preset", type=str, default="small", choices=sorted(_PRESETS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    run_llama4(torch.device("npu:0"), preset=args.preset, batch=args.batch,
               seq_len=args.seq_len, dtype=args.dtype, compile_model=args.compile,
               rtol=args.rtol, atol=args.atol)
