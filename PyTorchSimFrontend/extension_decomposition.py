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


@register_decomposition(aten.polar.default)
def decompose_polar(abs, angle):
    """`polar` as a VIEW over a real pair, so it never leaves the graph.

    Not a dtype problem like histc above -- a layout one, and the layout is
    decided by a hop that never reaches the CPU kernel. `aten::polar` is
    CompositeExplicitAutograd, so it is registered for THIS backend too and
    beats the boxed fallback: the composite runs here, allocates the result
    itself, and only then calls `aten::polar.out`, which is what the fallback
    sees. Measured, on a transposed input:

        aten::polar     out stride (1024, 32, 1)   contiguous
        aten::polar.out into a permuted out        (1024, 1, 32), kept
        cpu   torch.polar                          (1024, 1, 32)

    So the fallback is fine and the allocation is not. Inductor's meta models
    what a real backend does -- TensorIterator carries the operands'
    permutation into the output -- and gets

        assert_size_stride(buf14, (1, 32, 32), (1024, 1, 32),
                           'torch.ops.aten.polar.default')
        AssertionError: expected size 32==32, stride 32==1 at dim=1

    NOTHING ELSE ON THIS DEVICE LOSES THE PERMUTATION. mul, add and exp keep it
    (measured, npu against cpu, all three), and so do sinh and erfinv, which
    reach the fallback as functional ops and let the CPU kernel choose. It is
    the composite-with-a-fallback-out path alone.

    WHICH IS WHY THIS DOES NOT TRY TO FIX THE LAYOUT. Rewriting polar as
    `complex(abs*cos, abs*sin)` moved the identical assertion one op over, to
    `aten.complex.default` -- also composite, also allocating its own output.
    The answer is to stop having an extern call at all: build the pair with
    ops this backend compiles and let `view_as_complex` name it, which Inductor
    already treats as a view over real data. DeepSeek-V2's own RoPE proves the
    point -- its `view_as_complex` is fused into ordinary `*fp32` kernels
    (triton_npu_fused__unsafe_view_split_with_sizes_transpose_view_view_as_complex_5).

    `.contiguous()` IS LOAD-BEARING: view_as_complex wants the last dimension
    to have stride 1, and a stack over permuted operands does not
    ("RuntimeError: Tensor must have a last dimension with stride 1"). It is
    one real copy, on the device, in place of a CPU round trip.
    """
    pair = torch.stack((abs * torch.cos(angle), abs * torch.sin(angle)), dim=-1)
    return torch.view_as_complex(pair.contiguous())
