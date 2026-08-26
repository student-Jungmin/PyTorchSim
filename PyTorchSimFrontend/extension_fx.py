"""Which graphs Inductor is allowed to run this backend's post-grad passes on.

One process compiles for both devices, so the answer is read off the graph, and
it is asked where a pass is installed rather than inside the pass itself.
"""

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass


def is_npu_graph(g) -> bool:
    """Whether any tensor in this graph lives on the npu device."""
    graph = getattr(g, "graph", g)
    for node in graph.nodes:
        val = node.meta.get("val")
        for t in (val if isinstance(val, (tuple, list)) else (val,)):
            if isinstance(t, torch.Tensor) and t.device.type == "npu":
                return True
    return False


def npu_only(fn):
    """The same pass, a no-op on a graph with no npu tensor."""
    def _gated(graph):
        if is_npu_graph(graph):
            fn(graph)
    return _gated


class NpuOnlyPass(CustomGraphPass):
    """npu_only for a pass Inductor keys its cache on, whose uuid is kept."""

    def __init__(self, inner):
        self._inner = inner

    def uuid(self):
        return (self._inner.uuid(), "npu-only")

    def __call__(self, graph):
        if is_npu_graph(graph):
            self._inner(graph)
