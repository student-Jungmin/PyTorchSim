"""`topk` as k rounds of max-and-mask, so the selection stays in the graph.

A post-grad pass and not a decomposition: extension_counting_sort proves its
bound from `aten.topk.default` being in the post-grad graph, and a decomposition
removes it before that pass ever runs. Installed last so it expands topk only
after everything that reads its contract has.
"""

import torch
from torch import fx

aten = torch.ops.aten
prims = torch.ops.prims


def _spent(dtype, largest):
    """The value a taken element is replaced with, so it cannot win again."""
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    return info.min if largest else info.max


def _emit_topk(g, x, k, dim, largest, shape, dtype, dev):
    """Emit the k rounds and return the (values, indices) nodes."""
    call = g.call_function
    i64 = torch.int64
    n = int(shape[dim])
    seat_shape = [1] * len(shape)
    seat_shape[dim] = n

    rows = call(prims.iota.default, (n,),
                {"start": 0, "step": 1, "dtype": i64, "device": dev,
                 "requires_grad": False})
    ar = call(aten.view.default, (rows, seat_shape))
    seat = call(aten.expand.default, (ar, list(shape)))
    late = call(aten.full.default, (list(shape), n), {"dtype": i64, "device": dev})
    gone = call(aten.full.default, (list(shape), _spent(dtype, largest)),
                {"dtype": dtype, "device": dev})

    work, vals, idxs = x, [], []
    for _ in range(k):
        best = call(aten.amax.default if largest else aten.amin.default,
                    (work, [dim], True))
        cand = call(aten.where.self, (call(aten.eq.Tensor, (work, best)),
                                      seat, late))
        where = call(aten.amin.default, (cand, [dim], True))
        vals.append(best)
        idxs.append(where)
        work = call(aten.where.self,
                    (call(aten.eq.Tensor, (ar, where)), gone, work))
    return call(aten.cat.default, (vals, dim)), call(aten.cat.default, (idxs, dim))


def rewrite_topk(g) -> None:
    """Replace every `aten.topk.default` whose users are getitem with the rounds."""
    from operator import getitem
    if isinstance(g, fx.GraphModule):
        g = g.graph
    changed = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not aten.topk.default:
            continue
        users = list(node.users)
        if any(u.op != "call_function" or u.target is not getitem for u in users):
            continue
        x = node.args[0]
        xv = x.meta.get("val") if hasattr(x, "meta") else None
        if xv is None or xv.dim() == 0 or not xv.numel():
            continue
        k = int(node.args[1])
        dim = int(node.args[2]) if len(node.args) > 2 else -1
        dim = dim if dim >= 0 else dim + xv.dim()
        largest = bool(node.args[3]) if len(node.args) > 3 else True
        if k < 1 or k > int(xv.shape[dim]):
            continue
        with g.inserting_before(node):
            values, indices = _emit_topk(g, x, k, dim, largest, list(xv.shape),
                                         xv.dtype, xv.device)
        for u in users:
            u.replace_all_uses_with(values if u.args[1] == 0 else indices)
        changed += 1
    if changed:
        g.eliminate_dead_code()
        if g.owning_module is not None:
            g.owning_module.recompile()


def install():
    """Register the pass on Inductor's documented post-grad hook, chained last."""
    import torch._inductor.config as icfg
    prev = icfg.post_grad_custom_post_pass

    def _chained(g):
        if prev is not None:
            prev(g)
        rewrite_topk(g)

    icfg.post_grad_custom_post_pass = _chained
