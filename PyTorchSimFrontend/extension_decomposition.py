"""Decompositions that follow from what the DEVICE can do, not from codegen.

Both routes register a codegen backend for `npu` and only one of them is live in
a process (torch_openreg/__init__.py picks by TORCHSIM_TRITON_CODEGEN), so a
decomposition that belongs to one of them lives beside it -- mlir/ has its own.
This module is for the other kind: an op that has to be rewritten because of
what running on this device MEANS, which is the same fact whichever backend
emits the kernels.

An op with no `npu` kernel falls back, and the fallback runs on the CPU. So the
set of ops this device supports is ATen's CPU set, and where that set is
narrower than the CUDA one, a model written for CUDA arrives with a call this
device cannot make. That is a device-level statement and this is where it goes.
"""

import torch
from torch._inductor.decomposition import register_decomposition

aten = torch.ops.aten  # only for @register_decomposition target


@register_decomposition(aten.histc.default)
def decompose_histc(self, bins: int = 100, min: int = 0, max: int = 0):
    """`histc` of an integral tensor, which the CPU kernel does not implement.

        NotImplementedError: "histogram_cpu" not implemented for 'Int'

    Counting is dtype-blind, so a cast on the way in and back out is the whole
    of it. This is the same move transformers makes for itself one line above
    the call, and the reason it does not cover us is that it reads the DEVICE
    TYPE rather than asking what the device implements:

        histc_input = (expert_ids_g.float() if device.type in ("cpu", "mps")
                       else expert_ids_g.int())        # integrations/moe.py

    `npu` is neither "cpu" nor "mps", so the MoE router hands us the branch
    written for CUDA -- integral input -- and the fallback then runs on the very
    CPU kernel that the first branch exists to avoid. Returning the counts in
    the input's dtype keeps the CUDA branch's contract, which is the one the
    caller picked.

    FLOAT32 IS EXACT FOR THE VALUES THIS SEES and not for every integer: bin
    assignment goes through the float, so an input past 2^24 could land one bin
    over. What reaches it here are expert ids, bounded by the expert count.

    NOT A LOWERING ON PURPOSE. Counting is `(x == bin).sum()` over a broadcast
    against `arange(bins)`, one elementwise and one reduce, and it would keep
    the whole thing on the device -- worth doing when the CPU hop starts to
    matter. It also restates torch's bin-edge rounding, and a histogram that is
    one off routes tokens to the wrong expert with nothing to say so.
    """
    if self.is_floating_point():
        return NotImplemented
    counts = torch.histc(self.to(torch.float32), bins, min, max)
    return counts.to(self.dtype)
