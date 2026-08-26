"""Where this backend's post-grad passes may run, and the chain that holds them.

One process compiles for both devices, so the device is read off the graph. And
Inductor's post-grad hook is one slot per process, so passes join it through
install_once rather than each wrapping whatever it happened to find there.
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


def _identity(fn):
    """A name for a foreign callable that is the same in every process."""
    uuid = getattr(fn, "uuid", None)
    if callable(uuid):
        return uuid()
    module = getattr(fn, "__module__", "?")
    name = getattr(fn, "__qualname__", None)
    return f"{module}.{name}" if name else f"{module}.<unnamed>"


class _Chain(CustomGraphPass):
    """Inductor's single post-grad slot, as a chain that knows what is in it."""

    def __init__(self, foreign):
        self._foreign = foreign     # someone else's pass, kept and called first
        self._keys = []
        self._passes = []

    def add(self, key, fn) -> bool:
        """Append fn under key, or report that key is already chained."""
        if key in self._keys:
            return False
        self._keys.append(key)
        self._passes.append(fn)
        return True

    def uuid(self):
        head = () if self._foreign is None else (_identity(self._foreign),)
        return head + tuple(self._keys)

    def __call__(self, graph):
        if self._foreign is not None:
            self._foreign(graph)
        for fn in self._passes:
            fn(graph)


def install_once(key, fn):
    """Chain fn onto Inductor's post-grad hook, at most once per key."""
    from torch._inductor import config

    chain = config.post_grad_custom_post_pass
    if not isinstance(chain, _Chain):
        chain = _Chain(chain)
        config.post_grad_custom_post_pass = chain
    return chain.add(key, fn)
