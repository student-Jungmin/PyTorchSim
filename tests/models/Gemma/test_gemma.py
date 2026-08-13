"""Gemma 1 (2B and 7B) on the Triton codegen route.

Built from the ``transformers`` ``GemmaConfig`` with random weights, so it needs
no network and no checkpoint.  The model is ``transformers``' own
``GemmaForCausalLM`` -- nothing here reimplements the block, so what the test
exercises is the shipped module, not our reading of it.

A Gemma model is one block repeated (18 layers at 2B, 28 at 7B), so a preset
runs ONE layer.  What depth would add is the block-to-block seam, which the
Llama and GPT-2 tests already cross; what it would cost is a rerun of the same
kernels.

WHAT GEMMA 1 ADDS over the Llama/Qwen blocks already in this suite:

  * RMSNorm SCALED BY (1 + w).  ``GemmaRMSNorm`` holds a zero-initialised
    weight and applies ``x * (1 + w)``, upcasting to float32 first and casting
    back after -- every other RMSNorm in this suite is ``x * w``.  The offset
    is not cosmetic: with w near zero the Llama form collapses the activation
    and the Gemma form passes it through, so a backend that folded the two
    together would fail here and nowhere else.
  * EMBEDDING SCALED BY sqrt(hidden_size).  The embedding output is multiplied
    by a scalar (55.4 at 7B) before the first block.  A broadcast multiply of a
    Python-side constant against the full [batch, seq, hidden] activation.
  * head_dim 256 -- TWICE the largest in this suite, which tops out at 128
    (Llama 3, Qwen 2.5/3).  It doubles the reduction length inside attention
    and the tile the systolic array sees for q/k/v.
  * MQA at 2B: ``num_key_value_heads=1``, so n_rep = 8 and ONE kv head is
    broadcast across every query head.  The suite's GQA repeats are 1, 2, 3, 4,
    6, 7 and 8-into-8; a single kv head is the degenerate end none of them
    reach, and it makes k_proj/v_proj narrow (2048 -> 256) where q_proj is
    square.
  * GeGLU.  The gated MLP uses ``gelu_pytorch_tanh``; every gated MLP in this
    suite is SwiGLU (silu).  GPT-2 contributes a gelu, but an ungated one.
  * vocab 256000 with a tied lm_head -- the largest vocab in the suite.  The
    presets scale vocab down (it is a cost axis, not a character axis) but keep
    the tie, which is what makes the vocab projection share the embedding.

WHY 2B AND 7B ARE BOTH HERE.  They are not the same block at two widths:

  * 2B is MQA (kv_heads 1) and heads * head_dim == hidden (8 * 256 == 2048).
  * 7B is MHA (kv_heads 16) and heads * head_dim != hidden (16 * 256 == 4096
    against 3072 hidden), so q_proj is not square and o_proj shrinks 4096 ->
    3072.

Dropping either one drops a claim.  ``_assert_character`` cashes every claim
above against the built model, so a preset cannot pass while proving nothing.

    source /workspace/tnpu-env.sh
    python tests/models/Gemma/test_gemma.py --preset 7b

MEASURED, 2026-08-13, triton-npu pinned to develop 3434608, functional-only
config (Spike decides correctness; no cycles are reported):

    preset  params   kernels     max diff    time
    tiny    1.0M     15 / 18     9.54e-07     89s
    2b      126.9M   16 / 18     5.72e-06    207s
    7b      302.0M   15 / 17     7.63e-06    395s

"kernels" is unique / executed.  2b compiles one MORE kernel than 7b despite
being under half the size: MQA broadcasts its single kv head where 7B's MHA
does not, and that broadcast does not fuse into either neighbour.  Nothing here
was blocked when it was written -- Gemma 1 ran end to end on the first attempt
against this pin, so these numbers are a floor to regress against, not a fix.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# Presets shrink cost, not character.  Width carries head_dim, the GQA repeat
# and the GeGLU ratio, so width stays honest wherever it is affordable; depth
# and vocab are the cheap axes and are scaled down everywhere.  Both shipped
# Gemma 1 configs tie word embeddings, so every preset does too.
_PRESETS = {
    # Smoke.  Small enough to tell a broken test file from a broken backend.
    # Keeps MQA so the cheapest preset still covers the single-kv-head path.
    "tiny": dict(family="gemma", layers=1, hidden=256, heads=4, kv_heads=1,
                 head_dim=64, intermediate=1024, vocab=256, seq=16, tie=True),

    # Gemma-2B at its real width: MQA (n_rep = 8), head_dim 256, GeGLU 16384.
    "2b": dict(family="gemma", layers=1, hidden=2048, heads=8, kv_heads=1,
               head_dim=256, intermediate=16384, vocab=8192, seq=32, tie=True),

    # Gemma-7B at its real width: MHA, head_dim 256 decoupled from 3072 hidden,
    # GeGLU 24576.
    "7b": dict(family="gemma", layers=1, hidden=3072, heads=16, kv_heads=16,
               head_dim=256, intermediate=24576, vocab=8192, seq=32, tie=True),
}

_FAMILIES = {
    "gemma": ("transformers.models.gemma.configuration_gemma", "GemmaConfig",
              "transformers.models.gemma.modeling_gemma", "GemmaModel", "GemmaForCausalLM"),
}


def _dtype_from_str(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _import(module, name):
    import importlib

    return getattr(importlib.import_module(module), name)


def _build_config(preset, seq_len):
    p = _PRESETS[preset]
    cfg_mod, cfg_name, _, _, _ = _FAMILIES[p["family"]]
    Config = _import(cfg_mod, cfg_name)
    seq_len = seq_len if seq_len is not None else p["seq"]

    kwargs = dict(
        vocab_size=p["vocab"],
        hidden_size=p["hidden"],
        num_attention_heads=p["heads"],
        num_key_value_heads=p["kv_heads"],
        head_dim=p["head_dim"],
        intermediate_size=p["intermediate"],
        num_hidden_layers=p["layers"],
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        # GemmaMLP reads hidden_act; hidden_activation is the newer name that
        # shadows it.  Both are set to the same value so the gate cannot depend
        # on which one this transformers version happens to consult.
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        tie_word_embeddings=p["tie"],
        use_cache=False,
        # Eager keeps the graph free of the sdpa dispatch, so a failure is about
        # the model rather than about which attention kernel transformers picked.
        attn_implementation="eager",
    )
    return Config(**kwargs), seq_len


def _layer0(model):
    body = model.model if hasattr(model, "model") else model
    return body.layers[0]


def _assert_character(preset, cfg, model):
    """Fail if the model did not actually get the feature the preset claims.

    A preset is a claim about shape and about which Gemma the block is; this is
    where the claim is cashed.  Every check guards against a run that passes
    without exercising anything the suite does not already cover.
    """
    p = _PRESETS[preset]
    body = model.model if hasattr(model, "model") else model
    layer = _layer0(model)
    attn = layer.self_attn
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    # -- what makes a Gemma block not a Llama block ---------------------------
    # The (1 + w) offset.  Checked on the norm module rather than on the config,
    # because the config cannot express it: a zero weight that still passes the
    # activation through is only visible in the module's own arithmetic.
    norm = layer.input_layernorm
    assert type(norm).__name__ == "GemmaRMSNorm", \
        f"expected GemmaRMSNorm, got {type(norm).__name__}; the (1 + w) offset is the point of this test"
    probe = torch.ones(1, 1, cfg.hidden_size)
    with torch.no_grad():
        scaled = norm(probe)
    assert torch.allclose(scaled, probe, atol=1e-5), \
        ("GemmaRMSNorm with a zero weight must be the identity on a unit vector; "
         "a Llama-style x*w would return zeros here")

    assert attn.q_proj.bias is None, "Gemma has attention_bias=False; a bias here makes this a Qwen2 block"
    assert type(layer.mlp).__name__ == "GemmaMLP"
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(layer.mlp, name), f"GeGLU needs {name}; this MLP is not gated"
    act = type(layer.mlp.act_fn).__name__.lower()
    assert "gelu" in act or "tanh" in act, \
        f"the gated MLP must use a gelu, got {type(layer.mlp.act_fn).__name__}; silu here means this is a SwiGLU block"

    # head_dim 256 is the reason the attention tiles differ from the rest of the
    # suite, so a preset that quietly fell back to hidden/heads proves nothing.
    if preset != "tiny":
        assert head_dim == 256, f"Gemma 1 runs head_dim 256, got {head_dim}"
        assert head_dim == 2 * 128, "the suite's previous maximum was 128; this preset exists to double it"

    if preset == "2b":
        assert cfg.hidden_size == 2048 and cfg.intermediate_size == 16384, \
            "2b preset must run at the real Gemma-2B width"
        assert cfg.num_key_value_heads == 1, \
            f"Gemma-2B is MQA; kv_heads={cfg.num_key_value_heads} makes this an ordinary GQA preset"
        assert n_rep == 8, f"2b must broadcast one kv head across 8 query heads, got n_rep={n_rep}"
        assert attn.k_proj.out_features == head_dim, \
            f"MQA k_proj projects to a single head ({head_dim}), got {attn.k_proj.out_features}"
        assert cfg.num_attention_heads * head_dim == cfg.hidden_size, \
            "2B couples heads*head_dim to hidden; that coupling is what 7b breaks"

    if preset == "7b":
        assert cfg.hidden_size == 3072 and cfg.intermediate_size == 24576, \
            "7b preset must run at the real Gemma-7B width"
        assert n_rep == 1, f"Gemma-7B is MHA; got n_rep={n_rep}"
        assert cfg.num_attention_heads * head_dim != cfg.hidden_size, \
            ("this preset exists to decouple head_dim from hidden_size; "
             f"heads*head_dim == hidden ({cfg.hidden_size}) means it proves nothing new")
        assert attn.o_proj.in_features == 4096 and attn.o_proj.out_features == 3072, \
            "7b's o_proj must shrink 4096 -> 3072"

    if p["tie"]:
        head = getattr(model, "lm_head", None)
        assert head is not None, "tie check needs the LM head; run this preset with --part lm"
        assert head.weight.data_ptr() == body.embed_tokens.weight.data_ptr(), \
            "tie_word_embeddings was requested but lm_head does not share the embedding storage"


def _out(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type for comparison: {type(output)}")


@torch.no_grad()
def run_gemma(
    device,
    preset="tiny",
    part="lm",
    batch=1,
    seq_len=None,
    dtype="float32",
    compile_model=True,
    rtol=1e-2,
    atol=1e-2,
):
    p = _PRESETS[preset]
    _, _, mod_mod, body_name, lm_name = _FAMILIES[p["family"]]
    cls = _import(mod_mod, lm_name if part == "lm" else body_name)

    torch_dtype = _dtype_from_str(dtype)
    cfg, seq_len = _build_config(preset, seq_len)

    # Seed before construction: config-random weights otherwise differ per run,
    # so the worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    model_cpu = cls(cfg).to(dtype=torch_dtype).eval()

    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    print(f"preset={preset} family={p['family']} part={part} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"head_dim={head_dim} n_rep={n_rep} interm={cfg.intermediate_size} "
          f"vocab={cfg.vocab_size} seq={seq_len} dtype={dtype}")
    print(f"  gemma: act={cfg.hidden_act} embed_scale=sqrt({cfg.hidden_size})={cfg.hidden_size ** 0.5:.4f} "
          f"tie={cfg.tie_word_embeddings}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(preset, cfg, model_cpu)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), generator=g, dtype=torch.int64)

    # The explicit 4D causal mask is test_llama3x.py's, kept so a failure here is
    # comparable to that file's rather than to whatever _update_causal_mask
    # decided to build.
    min_dtype = torch.finfo(torch_dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=torch_dtype)
    if seq_len > 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1)

    cpu_out = _out(model_cpu(input_ids=input_ids, attention_mask=attn_mask))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _out(model_npu(input_ids=input_ids.to(device),
                             attention_mask=attn_mask.to(device)))

    test_result(f"Gemma {part} ({preset})", npu_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff > ", torch.max(torch.abs(npu_out.cpu() - cpu_out)))
    print("Gemma Simulation Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 1 (2B/7B) end to end on the Triton route")
    parser.add_argument("--preset", type=str, default="7b", choices=sorted(_PRESETS))
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body"],
                        help="lm = GemmaForCausalLM (adds the vocab projection); body = GemmaModel")
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

    run_gemma(
        torch.device("npu:0"),
        preset=args.preset,
        part=args.part,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        compile_model=args.compile,
        rtol=args.rtol,
        atol=args.atol,
    )
