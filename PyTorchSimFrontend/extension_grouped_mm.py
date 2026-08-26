"""transformers' grouped MoE matmul, rewritten with static shapes.

The op is opaque to torch.compile because its slice bounds come from a value,
so it leaves the graph and the expert matmuls run on the CPU.
"""

import torch
from torch import fx

from . import extension_fx

aten = torch.ops.aten
prims = torch.ops.prims


def _target():
    """The custom op's overload, or None until transformers has been imported."""
    ns = getattr(torch.ops, "transformers", None)
    op = getattr(ns, "grouped_mm_fallback", None) if ns is not None else None
    return op.default if op is not None else None


def _emit_dense_grouped_mm(g, inp, weight, offs, s, e, o, dtype, dev):
    """Every expert over the whole input, masked to its own rows and summed.

    Returns the (s, o) result node.
    """
    call = g.call_function
    i64 = torch.int64

    rows = call(prims.iota.default, (s,),
                {"start": 0, "step": 1, "dtype": i64, "device": dev,
                 "requires_grad": False})
    rows2 = call(aten.view.default, (rows, [s, 1]))
    offs64 = call(prims.convert_element_type.default, (offs, i64))
    offs2 = call(aten.view.default, (offs64, [1, e]))
    reached = call(aten.ge.Tensor, (rows2, offs2))
    reached64 = call(prims.convert_element_type.default, (reached, i64))
    expert_of = call(aten.sum.dim_IntList, (reached64, [1]))

    out = call(aten.full.default, ([s, o], 0), {"dtype": dtype, "device": dev})
    for i in range(e):
        wi = call(aten.clone.default, (call(aten.select.int, (weight, 0, i)),),
                  {"memory_format": torch.contiguous_format})
        part = call(aten.mm.default, (inp, wi))
        keep = call(aten.eq.Scalar, (expert_of, i))
        keep2 = call(aten.view.default, (keep, [s, 1]))
        keepf = call(prims.convert_element_type.default, (keep2, dtype))
        out = call(aten.add.Tensor, (out, call(aten.mul.Tensor, (part, keepf))))
    return out


def rewrite_grouped_mm(g) -> None:
    """Replace every `transformers.grouped_mm_fallback` call with the dense form."""
    target = _target()
    if target is None:
        return
    if isinstance(g, fx.GraphModule):
        g = g.graph
    changed = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not target:
            continue
        inp, weight, offs = node.args[0], node.args[1], node.args[2]
        iv = inp.meta.get("val") if hasattr(inp, "meta") else None
        wv = weight.meta.get("val") if hasattr(weight, "meta") else None
        if iv is None or wv is None or iv.dim() != 2 or wv.dim() != 3:
            continue
        s, e, o = int(iv.shape[0]), int(wv.shape[0]), int(wv.shape[2])
        with g.inserting_before(node):
            dense = _emit_dense_grouped_mm(g, inp, weight, offs, s, e, o,
                                           iv.dtype, iv.device)
        node.replace_all_uses_with(dense)
        changed += 1
    if changed:
        g.eliminate_dead_code()
        if g.owning_module is not None:
            g.owning_module.recompile()


def install():
    """Register the pass on Inductor's documented post-grad hook, once."""
    extension_fx.install_once("pytorchsim-grouped-mm", extension_fx.npu_only(rewrite_grouped_mm))
