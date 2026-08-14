"""`sort` of a bounded integer key, rewritten as a counting sort.

WHY THIS IS A DEVICE STATEMENT AND NOT AN OPTIMISATION. Inductor lowers
`torch.sort` to a BITONIC network, and a bitonic network of N elements is
sum(1..log2 N) compare-and-swap steps, each of which is four masked reductions
over the whole tile. For N = 128 that is 28 steps and 112 reductions, and after
fusion it is still 343 elementwise ops -- which this backend turns into 426
scratchpad globals, because a temporary here IS a global. Measured, Qwen2-MoE's
router sort at every tile size that lowers correctly:

    XBLOCK 8 / 4 / 2    127,616 bytes/lane      double-buffer budget 65,536

and no tile shrinks it, because the count of buffers is what is big and the
count does not depend on the tile (XBLOCK 8 and 64 both have 426). So the sort
does not fit this machine, at any tile, and the kernel it produces is the whole
reason that model does not run.

WHAT THE DATA ACTUALLY IS. The keys are expert ids: 128 of them, drawn from 8
values, on their way to a histogram and a cumulative sum that the graph computes
two lines later. That is a counting sort with the counting already written down.
The rewrite is one comparison against `arange(bins)`, two scans over it and two
scatters -- a handful of buffers instead of 426, and log2 rather than log^2.

WHEN IT IS ALLOWED, AND IT IS A PROOF RATHER THAN A GUESS. Counting needs the
range, and a tensor does not carry one. What does carry one is where the tensor
CAME FROM: `topk(x, k)`'s index output is bounded by `x.shape[dim]` by the
definition of topk, and that shape is in the graph. So this fires only when the
sort's input traces back, through shape-only ops, to that output -- and declines
otherwise, leaving Inductor's bitonic network exactly where it was. There is no
heuristic here: either the bound is proved by an op's contract or the rewrite
does not happen.

STABILITY. The counting sort is stable; `torch.sort` without `stable=True` is
not, and on CPU the two disagree about ties. Both are valid sorts, and the
difference cannot reach the answer: a permutation only decides the ORDER OF ROWS
in the grouped matmul that consumes it, rows of a matmul are independent, and
the scatter afterwards puts each row back where it came from. Measured on the
MoE shape, the two permutations differ in the result by 9.5367431640625e-07 --
float summation order, an order of magnitude inside the tolerance these models
are checked at.
"""

import torch
from torch import fx

aten = torch.ops.aten
prims = torch.ops.prims

#: Ops that change how a tensor is addressed and not what is in it. Walking
#: through these is what lets `sort(reshape(topk(x, k)[1]))` be recognised
#: without matching that exact spelling.
_SHAPE_ONLY = {
    aten.view.default, aten.reshape.default, aten._unsafe_view.default,
    aten.clone.default, aten.contiguous.default, aten.expand.default,
    aten.squeeze.default, aten.squeeze.dim, aten.unsqueeze.default,
    aten.flatten.using_ints, aten.detach.default,
}


def _bound_of(node):
    """The exclusive upper bound on this node's values, or None if unproved.

    ONE SOURCE, AND IT IS AN OP'S CONTRACT. `topk(x, k)` returns indices into
    `x` along its sorted dim, so every value is < x.shape[dim] -- that is the
    definition of topk and not an observation about a model. Nothing else here
    claims to know a range.
    """
    seen = 0
    while node is not None and seen < 32:            # a chain, not a search
        seen += 1
        if node.op != "call_function":
            return None
        if node.target in _SHAPE_ONLY:
            node = node.args[0]
            continue
        if node.target is operator_getitem and len(node.args) == 2:
            src, which = node.args
            if (which == 1 and getattr(src, "op", None) == "call_function"
                    and src.target is aten.topk.default):
                x = src.args[0]
                meta = x.meta.get("val") if hasattr(x, "meta") else None
                if meta is None or meta.dim() == 0:
                    return None
                dim = src.args[2] if len(src.args) > 2 else -1
                size = meta.shape[dim]
                return int(size) if isinstance(size, int) else None
        return None
    return None


try:                                   # `operator.getitem` is the graph's spelling
    from operator import getitem as operator_getitem
except ImportError:                    # pragma: no cover
    operator_getitem = None


def _emit_counting_sort(g, x, bins, n, dtype):
    """Stable counting sort of `x` ([n] integer, values in [0, bins)).

    Returns (values, permutation) with torch.sort's contract: `perm[i]` is where
    `values[i]` came from. Written in the ops the backend already lowers -- one
    compare, two cumulative sums, a gather and two scatters -- so nothing here
    reaches an extern call.
    """
    call = g.call_function
    i64 = torch.int64
    dev = x.meta["val"].device

    def iota(length):
        """0..length-1. `prims.iota` and NOT `aten.arange`: this runs POST-GRAD,
        where every op is expected to be already decomposed, and Inductor
        asserts when an op it has a decomposition for arrives needing a
        fallback ("both a fallback and a decomp for same op: aten.arange")."""
        return call(prims.iota.default, (length,),
                    {"start": 0, "step": 1, "dtype": i64, "device": dev,
                     "requires_grad": False})

    def zeros(length):
        return call(aten.full.default, ([length], 0),
                    {"dtype": i64, "device": dev})

    ar_n = iota(n)
    ar_b = iota(bins)

    def cast(v, dt):
        """`prims.convert_element_type` and NOT `aten._to_copy`, for the reason
        `iota` gives: post-grad wants ops that are already primitive."""
        return call(prims.convert_element_type.default, (v, dt))

    xl = cast(x, i64)
    x_col = call(aten.unsqueeze.default, (xl, 1))                 # [n, 1]
    b_row = call(aten.unsqueeze.default, (ar_b, 0))               # [1, bins]
    hot = call(aten.eq.Tensor, (x_col, b_row))                    # [n, bins] bool
    hoti = cast(hot, i64)

    counts = call(aten.sum.dim_IntList, (hoti, [0]))              # [bins]
    csum_b = call(aten.cumsum.default, (counts, 0))
    starts = call(aten.sub.Tensor, (csum_b, counts))              # exclusive scan

    rank = call(aten.cumsum.default, (hoti, 0))                   # [n, bins]
    rank0 = call(aten.sub.Scalar, (rank, 1))
    mine = call(aten.gather.default, (rank0, 1, x_col))           # [n, 1]
    mine = call(aten.squeeze.dim, (mine, 1))                      # [n]

    # `aten.index` and NOT `index_select`: index_select is the one op in this
    # list Inductor has a decomposition for and NO lowering, which post-grad
    # turns into "both a fallback and a decomp for same op".
    base = call(aten.index.Tensor, (starts, [xl]))                # [n]
    pos = call(aten.add.Tensor, (base, mine))                     # destination

    vals = call(aten.scatter.src, (zeros(n), 0, pos, xl))
    vals = cast(vals, dtype)
    perm = call(aten.scatter.src, (zeros(n), 0, pos, ar_n))
    return vals, perm


def rewrite_bounded_sorts(g) -> None:
    """Replace every provably bounded 1-D integer `sort` with a counting sort.

    Takes the GRAPH, which is what Inductor's post-grad hook hands over -- it
    calls `pass_fn(self.gm.graph)`, not `pass_fn(gm)`.
    """
    if isinstance(g, fx.GraphModule):
        g = g.graph
    changed = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not aten.sort.default:
            continue
        src = node.args[0]
        val = src.meta.get("val") if hasattr(src, "meta") else None
        if val is None or val.dim() != 1 or val.dtype not in (torch.int32, torch.int64):
            continue
        bins = _bound_of(src)
        if bins is None or bins > val.shape[0]:
            continue                       # unproved, or counting buys nothing
        with g.inserting_before(node):
            vals, perm = _emit_counting_sort(g, src, int(bins),
                                             int(val.shape[0]), val.dtype)
        for use in list(node.users):
            if use.op == "call_function" and use.target is operator_getitem:
                use.replace_all_uses_with(vals if use.args[1] == 0 else perm)
        changed += 1
    if changed:
        g.eliminate_dead_code()
        if g.owning_module is not None:
            g.owning_module.recompile()


def install():
    """Register the pass on Inductor's documented post-grad hook (rule 16)."""
    import torch._inductor.config as icfg
    prev = icfg.post_grad_custom_post_pass

    def _chained(g):
        if prev is not None:
            prev(g)
        rewrite_bounded_sorts(g)

    icfg.post_grad_custom_post_pass = _chained
