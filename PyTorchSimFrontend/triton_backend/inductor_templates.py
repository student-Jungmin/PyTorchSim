"""Let Inductor's mm/conv Triton templates reach this backend.

Without this they go to `extern_kernels.*`, which on npu either raises
`convolution_overrideable not implemented` or falls back to eager and simulates
nothing. The templates are not GPU-specific; `use_triton_template` is.
"""

import os

import torch

from PyTorchSimFrontend import extension_config

logger = extension_config.setup_logger()

_MIN_BLOCK = 16

_conv_groups = None

_NPU_CHOICES = None

_installed = False


def _register_npu_as_gpu():
    import torch._inductor.utils as inductor_utils

    if "npu" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("npu")


def _claim_triton_present():
    """has_triton() asks whether a supported DEVICE is available, not whether
    triton is installed. The missing piece is a driver we never use."""
    import torch._inductor.scheduler as scheduler
    import torch.utils._triton as triton_utils

    triton_utils.has_triton = lambda: True
    if hasattr(scheduler, "has_triton"):
        scheduler.has_triton = lambda: True


def _power_of_two(n):
    return n >= 1 and (n & (n - 1)) == 0


def _round_up_pow2(n):
    """The smallest legal Triton block extent that covers `n`."""
    v = _MIN_BLOCK
    while v < n:
        v *= 2
    return v


def _gemm_tiles(m, n, k, dtype_size):
    """This machine's mm tiles for [m, k] @ [k, n], best first.

    `gemm_tile_candidates` enumerates and ranks every tile that fits half the
    scratchpad. Torch's generic set is appended after, never before.
    """
    from torch._inductor.template_heuristics.triton import GemmConfig

    from .hardware import HardwareInfo

    tiles = HardwareInfo().gemm_tile_candidates(
        _round_up_pow2(int(m)), _round_up_pow2(int(n)), _round_up_pow2(int(k)),
        precision_bytes=int(dtype_size),
        n_prologue_node=2, n_prologue_extra_read=2,
        budget_divisor=2)

    out = []
    for tile_m, tile_n, tile_k in tiles:
        if not all(_power_of_two(b) and b >= _MIN_BLOCK
                   for b in (tile_m, tile_n, tile_k)):
            continue
        out.append(GemmConfig(tile_m, tile_n, tile_k, 1, 4))
    return out


def _register_template_heuristics():
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.template_heuristics.registry import (
        register_template_heuristic)
    from torch._inductor.template_heuristics.triton import (
        AddMMConfigMixin, BaseConfigHeuristic, MMTemplateConfigMixin)

    @register_template_heuristic(mm_template.uid, "npu")
    @register_template_heuristic(bmm_template.uid, "npu")
    class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, BaseConfigHeuristic):
        def _get_config_generator(self):
            """The hook the mixin documents for exactly this.

            It is the one place the shape is known, and a tile mapping blind to
            M, N and K is not one. _finalize_mm_configs still dedupes and clamps.
            """
            generic = super()._get_config_generator()

            def configs(m, n, k, **kwargs):
                from torch._inductor.virtualized import V
                try:
                    mnk = [int(V.graph.sizevars.size_hint(s)) for s in (m, n, k)]
                except Exception:
                    yield from generic(m, n, k, **kwargs)
                    return
                mapped = _gemm_tiles(*mnk, kwargs.get("dtype_size", 4))
                if not mapped:
                    logger.warning(
                        "[triton-npu] no mapped tile for %sx%sx%s is a legal "
                        "Triton block; falling back to the generic set", *mnk)
                yield from self._finalize_mm_configs(mapped)
                yield from generic(m, n, k, **kwargs)

            return configs

    @register_template_heuristic(mm_template.uid, "npu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")
    class NPUAddmmTemplateConfigHeuristic(AddMMConfigMixin,
                                          NPUMMTemplateConfigHeuristic):
        pass


def _groups_now():
    return getattr(_conv_groups, "value", 1) or 1


def _clamp_conv_block_n():
    """Tell the conv heuristic how many channels a GROUP has.

    `preprocess_mm_configs` narrows BLOCK_N to the extent it is handed, which for
    a grouped conv is all groups at once: mobilenet got 128 against GROUP_OUT_C 1.
    """
    import functools
    import inspect
    import threading

    global _conv_groups
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel

    _conv_groups = threading.local()
    sig = inspect.signature(conv_kernel.convolution)

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            try:
                groups = sig.bind(*args, **kwargs).arguments.get("groups", 1)
            except TypeError:
                groups = 1
            prev = getattr(_conv_groups, "value", 1)
            _conv_groups.value = groups if isinstance(groups, int) else 1
            try:
                return inner(*args, **kwargs)
            finally:
                _conv_groups.value = prev

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _lower_transposed_conv_as_a_direct_one():
    """Give a transposed convolution to the Triton template, as a direct one."""
    import functools
    import inspect

    from torch._inductor import ir
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.lowering import lowerings as L
    from torch._inductor.virtualized import V

    aten = torch.ops.aten
    prims = torch.ops.prims
    sig = inspect.signature(conv_kernel.convolution)

    def _per_dim(seq, n):
        seq = list(seq)
        return seq * n if len(seq) == 1 else seq

    def _spread(t, lead, extents, factor):
        """Put factor-1 zeros between neighbouring elements of each dim."""
        if all(f == 1 for f in factor):
            return t
        n = len(extents)
        paired = list(lead)
        for extent in extents:
            paired += [extent, 1]
        y = L[aten.view](t, paired)
        pad = []
        for i in reversed(range(n)):
            pad += [0, factor[i] - 1, 0, 0]
        y = L[aten.constant_pad_nd](y, pad, 0.0)
        y = L[aten.view](y, list(lead) +
                         [extents[i] * factor[i] for i in range(n)])
        for i in range(n):
            if factor[i] != 1:
                y = L[aten.slice](y, len(lead) + i, 0,
                                  extents[i] * factor[i] - (factor[i] - 1))
        return y

    def _rewrite(x, weight, stride, padding, dilation, output_padding, groups):
        """Return the (x, weight, *params) of the equivalent direct conv."""
        ints = V.graph.sizevars.guard_int_seq
        batch, in_chan, *spatial = ints(x.get_size())
        _, out_per_group, *kernel = ints(weight.get_size())
        n_sp = len(kernel)
        out_chan = out_per_group * groups
        stride = _per_dim(stride, n_sp)
        padding = _per_dim(padding, n_sp)
        dilation = _per_dim(dilation, n_sp)
        output_padding = _per_dim(output_padding, n_sp)

        y = _spread(x, [batch, in_chan], spatial, stride)

        pad = []
        for i in reversed(range(n_sp)):
            lo = dilation[i] * (kernel[i] - 1) - padding[i]
            pad += [lo, lo + output_padding[i]]
        if any(p != 0 for p in pad):
            y = L[aten.constant_pad_nd](y, pad, 0.0)

        w = L[aten.view](weight, [groups, in_chan // groups,
                                  out_chan // groups] + kernel)
        w = L[aten.permute](w, [0, 2, 1] + [3 + i for i in range(n_sp)])
        w = L[aten.view](w, [out_chan, in_chan // groups] + kernel)
        if any(k != 1 for k in kernel):
            w = L[prims.rev.default](w, [2 + i for i in range(n_sp)])

        w = _spread(w, [out_chan, in_chan // groups], kernel, dilation)

        return y, w, [1] * n_sp, [0] * n_sp, [1] * n_sp, [0] * n_sp

    def wrap(inner):
        @functools.wraps(inner)
        def convolution(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
            except TypeError:
                return inner(*args, **kwargs)

            x, weight = a["x"], a["weight"]
            if not a["transposed"] or ir.get_device_type(x) != "npu":
                return inner(*args, **kwargs)

            batchless = len(x.get_size()) == len(weight.get_size()) - 1
            if batchless:
                x = L[aten.expand](x, [1, *x.get_size()])

            y, w, stride, padding, dilation, output_padding = _rewrite(
                x, weight, a["stride"], a["padding"], a["dilation"],
                a["output_padding"], a["groups"])
            out = inner(y, w, a["bias"], stride, padding, dilation, False,
                        output_padding, a["groups"])
            return L[aten.squeeze](out, dim=0) if batchless else out

        return convolution

    packet = torch.ops.aten.convolution
    for key in [packet] + [getattr(packet, o) for o in packet.overloads()]:
        inner = inductor_lowering.lowerings.get(key)
        if inner is not None:
            inductor_lowering.lowerings[key] = wrap(inner)


def _size_conv_blocks_from_the_machine():
    """Offer conv tiles this machine has lanes for.

    `get_config_heuristics` has no registry lookup, so npu took a GPU's table.
    BLOCK_N is the lane count; M and K cost scratchpad and are offered ascending.
    """
    from torch._inductor.choices import InductorChoices
    from torch._inductor.template_heuristics.triton import (
        BaseConfigHeuristic, ConvConfig)

    from PyTorchSimFrontend import extension_config

    lanes = int(extension_config.vpu_num_lanes)

    class NPUConfigHeuristic(BaseConfigHeuristic):
        def __init__(self):
            super().__init__()
            self.conv_configs = [
                ConvConfig(64, lanes, 16, 1, 4),
                ConvConfig(32, lanes, 16, 1, 4),
                ConvConfig(64, lanes, 32, 1, 4),
            ]

        def get_conv_configs(self):
            base = super().get_conv_configs()

            def per_group(m, n, k, **kwargs):
                groups = _groups_now()
                return base(m, max(1, n // groups) if groups > 1 else n, k,
                            **kwargs)

            return per_group

    original = InductorChoices.get_config_heuristics

    def get_config_heuristics(self, device_type="cuda"):
        if device_type == "npu":
            return NPUConfigHeuristic()
        return original(self, device_type)

    InductorChoices.get_config_heuristics = get_config_heuristics


def _size_grouped_conv_grid_per_group():
    """Launch a grouped convolution over the channels a GROUP has.

    Inductor grids cdiv(OUT_C, BLOCK_N) on an axis the kernel masks against
    GROUP_OUT_C; the programs dropped are those whose stores were fully masked.
    """
    from torch._inductor.kernel import conv as conv_kernel
    from torch._inductor.select_algorithm import SymbolicGridFn

    @SymbolicGridFn
    def conv2d_grid(n, c, h, w, meta, *, cdiv):
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    @SymbolicGridFn
    def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
        groups = meta.get("GROUPS", 1) or 1
        return (
            cdiv(n * d * h * w, meta["BLOCK_M"]),
            cdiv(cdiv(c, groups), meta["BLOCK_N"]),
            groups,
        )

    conv_kernel.conv2d_grid = conv2d_grid
    conv_kernel.conv3d_grid = conv3d_grid
    for tmpl, fn in ((getattr(conv_kernel, "conv2d_template", None), conv2d_grid),
                     (getattr(conv_kernel, "conv3d_template", None), conv3d_grid)):
        if tmpl is not None:
            tmpl.grid = fn


def pick_config(choices):
    """Stand in for benchmarking: there is no device to time on, so the offered
    order wins. Extern ranks last, present only so a device with no registered
    heuristic (cpu) still has a choice."""
    from torch._inductor.select_algorithm import ExternKernelCaller

    return {c: (1e3 if isinstance(c, ExternKernelCaller) else 1.0) + i * 1e-3
            for i, c in enumerate(choices)}


def _short_circuit_degenerate_gemms():
    """A zero-length axis has no tile, so the heuristics offer no config and
    the empty choice list raises. A MoE expert routing no tokens gives
    [0, K] @ [K, N]."""
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.lowering import full, lowerings
    from torch._inductor.virtualized import V

    def wrap(op, bias):
        def wrapped(*args, _orig=lowerings[op], **kwargs):
            try:
                m, n, k, layout = mm_args(*args[bias:bias + 2],
                                          layout=kwargs.get("layout"))[:4]
                m, n, k = (int(V.graph.sizevars.size_hint(s)) for s in (m, n, k))
            except Exception:
                return _orig(*args, **kwargs)
            if m == 0 or n == 0 or (k == 0 and not bias):
                return full(layout.size, 0, dtype=layout.dtype,
                            device=layout.device)
            return _orig(*args, **kwargs)

        return wrapped

    aten = torch.ops.aten
    for op, bias in ((aten.mm, 0), (aten.bmm, 0),
                     (aten.addmm, 1), (aten.baddbmm, 1)):
        for name in op.overloads():
            o = getattr(op, name)
            if o in lowerings:
                lowerings[o] = wrap(o, bias)


def _npu_choices_class():
    global _NPU_CHOICES
    if _NPU_CHOICES is not None:
        return _NPU_CHOICES
    from torch._inductor.choices import InductorChoices

    from . import launch

    class NPUChoices(InductorChoices):
        @staticmethod
        def should_use_persistent_reduction(features, cooperative_reduction):
            base = InductorChoices.should_use_persistent_reduction(
                features, cooperative_reduction)
            if not base:
                return False
            try:
                extent = int(features.reduction_numel)
            except (TypeError, ValueError):
                return base
            if extent < 1:
                return base
            persistent = 1 << (extent - 1).bit_length()
            return persistent <= launch.reduction_block_for(extent)

    NPUChoices.__module__ = __name__
    NPUChoices.__qualname__ = "NPUChoices"
    globals()["NPUChoices"] = NPUChoices
    _NPU_CHOICES = NPUChoices
    return NPUChoices


def _decline_persistence_we_cannot_resize():
    """Persist a reduction only at a block this backend would have chosen.

    A persistent reduction bakes R0_BLOCK into its body, so the scratchpad retry
    halves a number nothing reads. Persist iff that block is the one we'd pin.
    """
    from torch._inductor import config

    if config.inductor_choices_class is None:
        config.inductor_choices_class = _npu_choices_class()


def _install_selection():
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        return pick_config(choices)

    AlgorithmSelectorCache.benchmark_choices = classmethod(benchmark_choices)
    AlgorithmSelectorCache.make_precompile_fn = lambda self, *a, **k: (lambda: None)


def install():
    """On by default; TORCHSIM_TRITON_TEMPLATES=0 opts out.

    Sending mm to aten simulates nothing, so a test that stops inside tnpu says
    more than one that passes without running the op.
    """
    global _installed
    if _installed or os.environ.get("TORCHSIM_TRITON_TEMPLATES", "1") == "0":
        return
    from torch._inductor import config

    _register_npu_as_gpu()
    _claim_triton_present()
    _register_template_heuristics()
    _lower_transposed_conv_as_a_direct_one()
    _size_conv_blocks_from_the_machine()
    _size_grouped_conv_grid_per_group()
    _clamp_conv_block_n()
    _short_circuit_degenerate_gemms()
    _decline_persistence_we_cannot_resize()
    _install_selection()

    config.max_autotune_gemm = True
    config.max_autotune_gemm_backends = "ATEN,TRITON"
    config.max_autotune_conv_backends = "ATEN,TRITON"
    config.triton.autotune_at_compile_time = False
    config.benchmark_epilogue_fusion = False

    _installed = True
