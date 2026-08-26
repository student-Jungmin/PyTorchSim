"""Rewrite complex subgraphs as real pairs, so Inductor can generate code.

Inductor does not lower complex arithmetic at all -- `unsupported_input_tensor`
in its lowering treats a complex tensor as one it can neither read nor write,
warns once ("Torchinductor does not support code generation for complex
operators"), and every complex op becomes an extern call. On this device an
extern call is the CPU fallback, so the work leaves the simulator entirely: the
kernels never exist, nothing is timed, and two of the ops involved are VIEWS,
which cannot fall back at all --

    The operator aten::view_as_complex appears to be a view operator, but it
    has no implementation for the backend "npu:0". View operators don't support
    falling back to run on the CPU, since the tensor's storage cannot be shared
    across devices.

A complex tensor is a real tensor with a trailing axis of 2, and `view_as_real`
says so in the graph already. So this pass keeps every complex value in that
form -- its PAIR -- and rewrites the ops that would have consumed the complex
one. Nothing is approximated: the pair is the same memory the complex tensor
would have had.

    view_as_complex(p)      the pair IS p
    view_as_real(z)         the pair IS what z was built from
    mul(z, w)               (ar*br - ai*bi, ar*bi + ai*br)
    mul(z, s)  s real       (ar*s, ai*s)

WHOLE OR NOTHING (rule 4's shape, borrowed). A component is rewritten only if
every complex value in it is produced and consumed by ops on that list. One
unhandled op and the pass leaves the graph exactly as it found it, because a
half-rewritten component is a graph with complex values whose producers are
gone.

MEASURED ON DeepSeek-V2-Lite, whose RoPE is the only complex path in the model
suite. Its graph, listed rather than guessed at:

    view_as_complex  x3     the two towers' pairs and freqs_cis
    unsqueeze        x1     freqs_cis broadcast to [1, 1, seq, dim/2]
    mul.Tensor       x2     the rotary multiply, once per tower
    view_as_real            the readers

The first version of this pass handled everything but the `unsqueeze` and so
refused the whole component -- correctly, and uselessly. V3 and V3.2 have no
complex path at all.
"""

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

from . import extension_fx

aten = torch.ops.aten

#: Ops this pass knows how to say in real arithmetic. Everything else makes the
#: component ineligible -- see the docstring.
_MUL = (aten.mul.Tensor, aten.mul.Scalar)

#: Shape ops that leave the pair's trailing axis alone, so the same op says the
#: same thing about the pair. Only `unsqueeze` is here because only `unsqueeze`
#: was measured on a complex value (DeepSeek-V2 broadcasts freqs_cis with it);
#: view, expand and the rest bail out rather than be written untested.
_SHAPE = (aten.unsqueeze.default,)

_HANDLED = (aten.view_as_complex.default,) + _MUL + _SHAPE


def _val(node):
    return node.meta.get("val", None) if isinstance(node, torch.fx.Node) else node


def _is_complex(node):
    v = _val(node)
    return isinstance(v, torch.Tensor) and v.is_complex()


def _fake_mode(graph):
    for n in graph.nodes:
        v = _val(n)
        if isinstance(v, torch.Tensor) and hasattr(v, "fake_mode"):
            return v.fake_mode
    return None


class ComplexToRealPairs(CustomGraphPass):
    """Post-grad pass: complex values become their real [..., 2] pairs."""

    def uuid(self):
        # Part of Inductor's cache key. Bump it when the rewrite changes, or a
        # graph compiled by the old one is replayed for the new.
        return "pytorchsim-complex-to-real-pairs-2"

    def __call__(self, graph: torch.fx.Graph) -> None:
        complex_nodes = [n for n in graph.nodes if _is_complex(n)]
        if not complex_nodes:
            return

        # ELIGIBILITY, ASKED BEFORE ANYTHING IS WRITTEN. Every complex value has
        # to be produced by an op we can restate, and read only by ops we can
        # restate. A complex graph OUTPUT is out too: the caller asked for a
        # complex tensor and a pair is not one.
        for n in complex_nodes:
            if n.op != "call_function":
                return
            if n.target not in _HANDLED:
                return
            for user in n.users:
                if user.op == "output":
                    return
                if user.target is aten.view_as_real.default:
                    continue
                if user.target in _HANDLED:
                    continue
                return

        fake_mode = _fake_mode(graph)

        def _unwrap(a):
            # `cat` takes a LIST of nodes, so the meta computation has to walk
            # into it -- `_val` of a list is that list, which is a list of Nodes
            # and not of tensors ("Expected a value of type 'List[Tensor]'").
            if isinstance(a, (list, tuple)):
                return type(a)(_unwrap(x) for x in a)
            return _val(a)

        def call(target, *args):
            node = graph.call_function(target, args)
            if fake_mode is not None:
                with fake_mode:
                    node.meta["val"] = target(*[_unwrap(a) for a in args])
            return node

        pair = {}  # complex node -> the node holding its [..., 2] real pair

        for n in complex_nodes:
            with graph.inserting_before(n):
                if n.target is aten.view_as_complex.default:
                    # The pair is the operand, unchanged. No node is created:
                    # this is the whole reason the rewrite costs nothing.
                    pair[n] = n.args[0]
                    continue

                if n.target in _SHAPE:
                    # The pair carries one extra axis, at the END. A
                    # non-negative dim counts from the front and means the same
                    # thing; a negative one counts from the back and has to
                    # step over that axis.
                    dim = n.args[1]
                    pair[n] = call(n.target, pair[n.args[0]],
                                   dim if dim >= 0 else dim - 1)
                    continue

                a, b = n.args[0], n.args[1]
                if _is_complex(a) and _is_complex(b):
                    pa, pb = pair[a], pair[b]
                    ar = call(aten.select.int, pa, -1, 0)
                    ai = call(aten.select.int, pa, -1, 1)
                    br = call(aten.select.int, pb, -1, 0)
                    bi = call(aten.select.int, pb, -1, 1)
                    re = call(aten.sub.Tensor, call(aten.mul.Tensor, ar, br),
                              call(aten.mul.Tensor, ai, bi))
                    im = call(aten.add.Tensor, call(aten.mul.Tensor, ar, bi),
                              call(aten.mul.Tensor, ai, br))
                    pair[n] = call(aten.cat.default,
                                   [call(aten.unsqueeze.default, re, -1),
                                    call(aten.unsqueeze.default, im, -1)], -1)
                else:
                    # One side is real -- a scalar or a tensor that broadcasts
                    # against the complex one. It scales both components, and
                    # the pair's trailing axis is the one it must NOT reach, so
                    # a tensor operand gets that axis added.
                    cx, other = (a, b) if _is_complex(a) else (b, a)
                    if isinstance(other, torch.fx.Node):
                        other = call(aten.unsqueeze.default, other, -1)
                    pair[n] = call(aten.mul.Tensor, pair[cx], other)

        # The readers. `view_as_real` asked for exactly the pair, and a complex
        # reader has already been given one above.
        for n in complex_nodes:
            for user in list(n.users):
                if user.target is aten.view_as_real.default:
                    user.replace_all_uses_with(pair[n])

        graph.eliminate_dead_code()


_PASS = ComplexToRealPairs()


def install():
    """Put the pass on Inductor's post-grad hook, keeping anyone else's.

    The hook holds one callable, so a second consumer of the same standard
    extension point would silently replace this one (or be replaced by it).
    Chaining is the only version of "install" that is safe to call from an
    import, and install_once is what makes calling it twice harmless.
    """
    extension_fx.install_once(_PASS.uuid(), extension_fx.npu_only(_PASS))


install()
