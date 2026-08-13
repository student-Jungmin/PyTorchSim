"""Gemma 1, 2 and 3 (text) on the Triton codegen route.

Built from the ``transformers`` configs with random weights, so it needs no
network and no checkpoint.  The models are ``transformers``' own
``GemmaForCausalLM`` / ``Gemma2ForCausalLM`` / ``Gemma3ForCausalLM`` -- nothing
here reimplements a block, so what the test exercises is the shipped module,
not our reading of it.  One file covers all three generations because they are
deltas on one block, the way the Qwen file covers its family.

HOW DEEP A PRESET RUNS IS DECIDED BY THE REPEATING UNIT, not by taste:

  * Gemma 1's unit is ONE layer -- 18 of them at 2B, 28 at 7B, all identical.
    A preset runs one.  What depth would add is the block-to-block seam, which
    the Llama and GPT-2 tests already cross.
  * Gemma 2 alternates local and global attention, ``is_sliding = not
    (layer_idx % 2)``, so its unit is TWO layers.  A one-layer preset would run
    the local variant and never the global one.
  * Gemma 3 runs FIVE local layers to one global, ``is_sliding =
    (layer_idx + 1) % sliding_window_pattern`` with the pattern at 6, so its
    unit is SIX layers.  That is affordable only because Gemma 3's smallest
    shipped text model is narrow (1152 hidden), which is why ``3-1b`` is the
    preset that carries the pattern and ``3-4b`` carries width alone.

WHAT EACH GENERATION ADDS, and why it needs its own preset:

  * 1  the baseline block -- see the Gemma-1 section below.
  * 2  LOCAL/GLOBAL ALTERNATION and LOGIT SOFT-CAPPING.  Attention logits are
       capped at 50 and the final logits at 30, each as
       ``tanh(x / cap) * cap`` -- a tanh on the [batch, heads, seq, seq] score
       tensor and again on [batch, seq, vocab].  Nothing else in this suite
       caps.  Gemma 2 also adds a SECOND PAIR of RMSNorms per layer
       (``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``, on top
       of input/post_attention), so the block carries four norms where Llama
       and Gemma 1 carry two.
  * 3  TWO ROPE BASES IN ONE MODEL.  Local layers use ``rope_local_base_freq``
       (10000) and global layers ``rope_theta`` (1000000), materialised as two
       separate rotary modules (``rotary_emb`` and ``rotary_emb_local``) whose
       cos/sin the layers pick between.  Every other model here has exactly one
       inv_freq.  Soft-capping is gone again (both caps are None).  Gemma 3
       also applies q_norm/k_norm over head_dim -- that is NOT new, Qwen 3
       contributes it -- so the presets do not claim it.

THE SLIDING WINDOW IS SCALED DOWN WITH seq, AND THAT IS LOAD-BEARING.  Both
shipped configs use a 4096-token window; at seq 32 ``torch.tril(..., diagonal=
-4096)`` selects NOTHING, so the local layers become byte-identical to the
global ones and the alternation the presets exist to test evaporates.  Measured:
window 4096 at seq 32 masks 0 elements, window 8 masks 300.  So window is a
preset field, ``_assert_character`` refuses any preset whose window is not
strictly inside its seq, and a "Gemma 2 passes" that never built a band mask
cannot happen quietly.

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
config (Spike decides correctness; no cycles are reported).  Every preset ran
with its OWN TORCHSIM_DUMP_PATH: the kernel cache is keyed on Triton source
text alone, and a bmm template's text is identical for two presets that differ
only in head COUNT, so a shared dump lets one preset's kernel answer for
another's.

    preset         gen  layers  params   kernels    max diff    time
    tiny            1      1      1.0M    15 /  18  9.5367e-07    73s
    2b              1      1    126.9M    16 /  18  5.7220e-06   269s
    7b              1      1    302.0M    15 /  17  7.6294e-06   394s
    2-27b-scalar    2      1     11.5M    20 /  22  3.8147e-06   101s
    2-2b            2      2    174.6M    24 /  38  5.7220e-06   317s
    3-4b            3      1    115.4M    23 /  24  5.7220e-06   226s
    3-1b            3      6    170.5M    25 / 100  4.5896e-06   487s

"kernels" is unique / executed.  Three things the counts say:

  * GEMMA 2 COSTS KERNELS, NOT WIDTH.  2-27b-scalar compiles 20 unique kernels
    at 11.5M parameters where Gemma 1's 302M 7b compiles 15.  The extra work is
    per-layer and shape-independent: two more RMSNorms and two tanh soft-caps.
  * LOCAL AND GLOBAL ATTENTION COMPILE SEPARATELY.  2-2b runs two layers and
    lands at 24 unique against 38 executed -- the pair does not share one
    attention kernel, which is the observable consequence of the alternation.
  * DEPTH REUSES.  3-1b executes 100 kernels for 25 unique across six layers:
    the five local layers share, and the global layer's different rope base
    earns it its own.  That is the argument for running exactly one period and
    not two.

2b compiles one MORE kernel than 7b despite being under half the size: MQA
broadcasts its single kv head where 7B's MHA does not, and that broadcast does
not fuse into either neighbour.

Nothing here was blocked when it was written -- all three generations ran end to
end on the first attempt against this pin, with no backend change.  These
numbers are a floor to regress against, not a fix.
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

    # Gemma2-2B at its real width, TWO layers -- one local, one global, which is
    # Gemma 2's whole repeating unit.  Soft-capping on at its shipped values.
    "2-2b": dict(family="gemma2", layers=2, hidden=2304, heads=8, kv_heads=4,
                 head_dim=256, intermediate=9216, vocab=8192, seq=32, tie=True,
                 window=8, attn_softcap=50.0, final_softcap=30.0, qpas=256),

    # Gemma2-27B's attention scaling, at a width that costs nothing: 27B is the
    # one Gemma whose query_pre_attn_scalar (144) is NOT head_dim (128), so the
    # pre-attention scale is not head_dim**-0.5.  One layer -- this preset is a
    # claim about a scalar, and the alternation is 2-2b's job.
    "2-27b-scalar": dict(family="gemma2", layers=1, hidden=1024, heads=8, kv_heads=4,
                         head_dim=128, intermediate=2048, vocab=2048, seq=32, tie=True,
                         window=8, attn_softcap=50.0, final_softcap=30.0, qpas=144),

    # Gemma3-1B at its real width, SIX layers -- one full 5:1 period, so five
    # local layers and the single global one all run.  Affordable only because
    # 1B is narrow: 1152 hidden with MQA keeps six layers near 170M.
    "3-1b": dict(family="gemma3", layers=6, hidden=1152, heads=4, kv_heads=1,
                 head_dim=256, intermediate=6912, vocab=8192, seq=32, tie=True,
                 window=8, rope_local=10000.0, rope_global=1000000.0, pattern=6, qpas=256),

    # Gemma3-4B's width, one layer.  A width claim only: at pattern 6 layer 0 is
    # local, so this preset says nothing about the period -- 3-1b covers that.
    "3-4b": dict(family="gemma3", layers=1, hidden=2560, heads=8, kv_heads=4,
                 head_dim=256, intermediate=10240, vocab=8192, seq=32, tie=True,
                 window=8, rope_local=10000.0, rope_global=1000000.0, pattern=6, qpas=256),
}

_FAMILIES = {
    "gemma": ("transformers.models.gemma.configuration_gemma", "GemmaConfig",
              "transformers.models.gemma.modeling_gemma", "GemmaModel", "GemmaForCausalLM"),
    "gemma2": ("transformers.models.gemma2.configuration_gemma2", "Gemma2Config",
               "transformers.models.gemma2.modeling_gemma2", "Gemma2Model", "Gemma2ForCausalLM"),
    "gemma3": ("transformers.models.gemma3.configuration_gemma3", "Gemma3TextConfig",
               "transformers.models.gemma3.modeling_gemma3", "Gemma3TextModel", "Gemma3ForCausalLM"),
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
        rms_norm_eps=1e-6,
        # Gemma 1's MLP reads hidden_act; 2 and 3 read hidden_activation.  Both
        # are set so the gate cannot depend on which name a given transformers
        # version happens to consult.
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        tie_word_embeddings=p["tie"],
        use_cache=False,
        # Eager keeps the graph free of the sdpa dispatch, so a failure is about
        # the model rather than about which attention kernel transformers picked.
        attn_implementation="eager",
    )

    if p["family"] == "gemma":
        kwargs["rope_theta"] = 10000.0
    else:
        # The window is shrunk to fit inside seq on purpose; see the module
        # docstring.  _assert_character refuses a preset where it does not.
        kwargs["sliding_window"] = p["window"]
        kwargs["query_pre_attn_scalar"] = p["qpas"]
        # Both 2 and 3 default this to "hybrid", which pairs a static cache with
        # the window for generate().  use_cache is off and this test never
        # generates, so leaving it set only produces a warning that does not
        # describe the run.
        kwargs["cache_implementation"] = None

    if p["family"] == "gemma2":
        kwargs["rope_theta"] = 10000.0
        kwargs["attn_logit_softcapping"] = p["attn_softcap"]
        kwargs["final_logit_softcapping"] = p["final_softcap"]

    if p["family"] == "gemma3":
        kwargs["rope_theta"] = p["rope_global"]
        kwargs["rope_local_base_freq"] = p["rope_local"]
        kwargs["sliding_window_pattern"] = p["pattern"]
        # Gemma 3 dropped both caps; pinning them to None keeps a future
        # transformers default from quietly reintroducing a tanh.
        kwargs["attn_logit_softcapping"] = None
        kwargs["final_logit_softcapping"] = None

    return Config(**kwargs), seq_len


def _layer0(model):
    body = model.model if hasattr(model, "model") else model
    return body.layers[0]


def _assert_character(preset, cfg, model, seq_len):
    """Fail if the model did not actually get the feature the preset claims.

    A preset is a claim about shape and about which Gemma the block is; this is
    where the claim is cashed.  Every check guards against a run that passes
    without exercising anything the suite does not already cover.
    """
    p = _PRESETS[preset]
    family = p["family"]
    body = model.model if hasattr(model, "model") else model
    layer = _layer0(model)
    attn = layer.self_attn
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    # -- what makes any Gemma block not a Llama block -------------------------
    # The (1 + w) offset, shared by all three generations.  Checked on the norm
    # module rather than on the config, because the config cannot express it: a
    # zero weight that still passes the activation through is only visible in
    # the module's own arithmetic.
    norm = layer.input_layernorm
    norm_name = type(norm).__name__
    assert norm_name.startswith("Gemma") and norm_name.endswith("RMSNorm"), \
        f"expected a Gemma*RMSNorm, got {norm_name}; the (1 + w) offset is the point of this test"
    probe = torch.ones(1, 1, cfg.hidden_size)
    with torch.no_grad():
        scaled = norm(probe)
    assert torch.allclose(scaled, probe, atol=1e-5), \
        (f"{norm_name} with a zero weight must be the identity on a unit vector; "
         "a Llama-style x*w would return zeros here")

    assert attn.q_proj.bias is None, "Gemma has attention_bias=False; a bias here makes this a Qwen2 block"
    assert type(layer.mlp).__name__.endswith("MLP")
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(layer.mlp, name), f"GeGLU needs {name}; this MLP is not gated"
    act = type(layer.mlp.act_fn).__name__.lower()
    assert "gelu" in act or "tanh" in act, \
        f"the gated MLP must use a gelu, got {type(layer.mlp.act_fn).__name__}; silu here means this is a SwiGLU block"

    # head_dim 256 is the reason the attention tiles differ from the rest of the
    # suite, so a preset that quietly fell back to hidden/heads proves nothing.
    # 2-27b-scalar is the deliberate exception: 27B is the one Gemma at 128.
    if preset not in ("tiny", "2-27b-scalar"):
        assert head_dim == 256, f"this preset runs head_dim 256, got {head_dim}"
        assert head_dim == 2 * 128, "the suite's previous maximum was 128; this preset exists to double it"

    # -- Gemma 1 is the plain block: no window, no caps -----------------------
    if family == "gemma":
        assert getattr(cfg, "sliding_window", None) is None, \
            "Gemma 1 has no sliding window; a window here means the wrong config class was built"
        assert getattr(cfg, "attn_logit_softcapping", None) is None, \
            "Gemma 1 does not soft-cap; capping here means this is a Gemma 2 block"

    # -- the window must actually bite, or the alternation proves nothing -----
    if family in ("gemma2", "gemma3"):
        window = cfg.sliding_window
        assert window is not None and window < seq_len, \
            (f"sliding_window ({window}) must be strictly inside seq ({seq_len}); "
             "at the shipped 4096 against seq 32 the band mask selects nothing and "
             "the local layers become byte-identical to the global ones")
        band = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-window)
        assert band.any(), \
            f"window {window} at seq {seq_len} masks no element; this preset would test causal attention twice"

        # is_sliding lives on the decoder layer in both generations; Gemma 3
        # also mirrors it onto the attention, Gemma 2 does not.
        sliding = [bool(l.is_sliding) for l in body.layers]
        assert len(sliding) == cfg.num_hidden_layers
        assert cfg.query_pre_attn_scalar == p["qpas"], \
            f"query_pre_attn_scalar must be {p['qpas']}, got {cfg.query_pre_attn_scalar}"

    # -- Gemma 2: alternation, soft-capping, four norms per layer -------------
    if family == "gemma2":
        assert cfg.attn_logit_softcapping == p["attn_softcap"] and \
            cfg.final_logit_softcapping == p["final_softcap"], \
            "Gemma 2 soft-caps attention and final logits; without both this is a Gemma 1 block with a window"
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            assert hasattr(layer, name), \
                f"Gemma 2 carries four RMSNorms per layer; {name} is missing"
        if cfg.num_hidden_layers >= 2:
            assert sliding[:2] == [True, False], \
                f"Gemma 2 alternates local then global; got {sliding[:2]}"

    # -- Gemma 3: two rope bases, 5:1 period, no caps -------------------------
    if family == "gemma3":
        assert cfg.attn_logit_softcapping is None and cfg.final_logit_softcapping is None, \
            "Gemma 3 dropped soft-capping; a cap here means the preset is really a Gemma 2"
        assert hasattr(body, "rotary_emb") and hasattr(body, "rotary_emb_local"), \
            "Gemma 3 needs both a global and a local rotary; one of them is missing"
        glob = body.rotary_emb.inv_freq
        loc = body.rotary_emb_local.inv_freq
        assert not torch.allclose(glob, loc), \
            ("the two rope bases must differ -- that is the whole point of the Gemma 3 preset; "
             f"rope_theta={cfg.rope_theta} rope_local_base_freq={cfg.rope_local_base_freq}")
        if cfg.num_hidden_layers >= p["pattern"]:
            assert sliding.count(False) == cfg.num_hidden_layers // p["pattern"], \
                f"one layer in {p['pattern']} must be global; got {sliding}"
            assert sliding[p["pattern"] - 1] is False, \
                f"the global layer sits last in the period; got {sliding}"
        else:
            assert all(sliding), \
                ("a preset shorter than the period runs local layers only; "
                 "if a global one appeared here the period changed")

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

    if preset == "2-2b":
        assert cfg.hidden_size == 2304 and cfg.intermediate_size == 9216, \
            "2-2b preset must run at the real Gemma2-2B width"
        assert n_rep == 2, f"Gemma2-2B is GQA 8/4; got n_rep={n_rep}"
        assert cfg.num_hidden_layers == 2, \
            ("Gemma 2's repeating unit is a local/global PAIR; one layer here would "
             "run the local variant and never the global one")

    if preset == "2-27b-scalar":
        assert cfg.query_pre_attn_scalar != head_dim, \
            ("this preset exists because 27B scales queries by 144 while its head_dim is 128; "
             f"scalar == head_dim ({head_dim}) means it proves nothing the other presets do not")

    if preset == "3-1b":
        assert cfg.hidden_size == 1152 and cfg.intermediate_size == 6912, \
            "3-1b preset must run at the real Gemma3-1B width"
        assert cfg.num_key_value_heads == 1 and n_rep == 4, \
            f"Gemma3-1B is MQA with 4 query heads; got kv={cfg.num_key_value_heads} n_rep={n_rep}"
        assert cfg.num_hidden_layers == p["pattern"], \
            (f"3-1b runs one full {p['pattern']}-layer period so the single global layer runs; "
             f"got {cfg.num_hidden_layers} layers")

    if preset == "3-4b":
        assert cfg.hidden_size == 2560 and cfg.intermediate_size == 10240, \
            "3-4b preset must run at the real Gemma3-4B width"
        assert cfg.num_hidden_layers < p["pattern"], \
            "3-4b is a width claim; if it ran a full period it would duplicate 3-1b's job at higher cost"

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
    act = getattr(cfg, "hidden_activation", None) or getattr(cfg, "hidden_act", None)
    print(f"  gemma: act={act} embed_scale=sqrt({cfg.hidden_size})={cfg.hidden_size ** 0.5:.4f} "
          f"tie={cfg.tie_word_embeddings}")
    if p["family"] in ("gemma2", "gemma3"):
        body_cpu = model_cpu.model if hasattr(model_cpu, "model") else model_cpu
        sliding = ["local" if l.is_sliding else "GLOBAL" for l in body_cpu.layers]
        print(f"  window={cfg.sliding_window} (seq={seq_len}) qpas={cfg.query_pre_attn_scalar} "
              f"layers={'/'.join(sliding)}")
        if p["family"] == "gemma2":
            print(f"  softcap: attn={cfg.attn_logit_softcapping} final={cfg.final_logit_softcapping}")
        else:
            print(f"  rope: global={cfg.rope_theta} local={cfg.rope_local_base_freq} "
                  f"pattern={cfg.sliding_window_pattern}")
    print("model params:", sum(x.numel() for x in model_cpu.parameters()))

    _assert_character(preset, cfg, model_cpu, seq_len)

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


# One preset per generation, because a generation is what this file adds and a
# gate that skipped one would leave it claimed-but-unrun.  7b is Gemma 1 at its
# real width, 2-2b is the Gemma 2 local/global pair, 3-1b is one full Gemma 3
# period.
#
# MEASURED AS CI RUNS IT -- all three in ONE process, one dump path: 1137s and
# 64 unique kernels, against the sweep's 1800s timeout.  Run separately they sum
# to 1198s, so the three presets share a little codegen rather than colliding in
# it: each preset's max diff is identical to its isolated run (7.6294e-06,
# 5.7220e-06, 4.5896e-06), which is what rules out the shared-dump hazard that
# the per-preset numbers in the docstring were measured to avoid.
_GATE = ("7b", "2-2b", "3-1b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 1/2/3 end to end on the Triton route")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(_PRESETS),
                        help=f"one preset; the default runs the gate ({', '.join(_GATE)})")
    parser.add_argument("--part", type=str, default="lm", choices=["lm", "body"],
                        help="lm = ...ForCausalLM (adds the vocab projection); body = ...Model")
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

    for preset in ([args.preset] if args.preset else _GATE):
        run_gemma(
            torch.device("npu:0"),
            preset=preset,
            part=args.part,
            batch=args.batch,
            seq_len=args.seq_len,
            dtype=args.dtype,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
