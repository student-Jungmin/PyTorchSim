"""YOLO v3 / v5 / v6 / v8 / v9 / v10 / 11 / 12 / 26, on the Triton codegen route.

One file per family, and here the family is defined by a PARSER rather than by a
checkpoint.  ``ultralytics.nn.tasks.parse_model`` turns a version's YAML into a
``nn.Sequential``, so every version in this file is built the same way -- from
``ultralytics/cfg/models/<version>/*.yaml`` with random weights, no ``.pt``, no
network, no torch.hub -- and what changes between presets is which blocks the
parser instantiates.  That is the axis worth covering: the backbone/neck blocks
are what each YOLO generation actually changed.

``tests/models/Yolov5/test_yolov5.py`` stays as it is and covers something this
file cannot: real pretrained weights, fetched through torch.hub from the
standalone ultralytics/yolov5 repository, which has its own parser AND its own
anchor-based head.  This file is the config-built counterpart across
generations.

THE PARSER NORMALISES THE HEAD, which is the first thing to know about this
file and was NOT the assumption it was written under.  ``ultralytics/cfg/models``
carries a v3 and a v5 YAML, but they terminate in the modern ``Detect``:
anchor-free, decoupled, ``reg_max=16``, DFL.  Measured, not assumed -- every
preset below reports ``reg_max=16`` and a real ``DFL`` module except 26, which
sets ``reg_max=1`` and an ``Identity``.  So the ANCHOR-BASED head is not
reachable through this parser at all, and the hub test is the only place in the
suite it runs.  What a preset here claims is therefore its BACKBONE and NECK,
plus the two head deltas the parser does express: end2end (v10, 26) and the
dropped DFL (26).

WHAT EACH VERSION ADDS, and why it needs its own preset:

  * v3     The pre-CSP baseline, and the only preset with no C-block of any
           kind: ``yolov3-tiny`` is Conv + ``MaxPool2d`` all the way down, and
           it is the only model in this suite that emits ``ZeroPad2d`` -- the
           asymmetric (0,1,0,1) pad in front of its last pool, which is the one
           pool it runs at stride 1 rather than 2.
  * v5     ``C3`` -- a CSP block whose two branches meet at a concat -- plus
           ``SPPF``, the serial 5x5 maxpool pyramid that replaced SPP's
           parallel one, and the plain ``Bottleneck`` nested inside C3.
  * v6     ``ConvTranspose2d`` in the neck: every other preset upsamples with
           ``nn.Upsample``, and this is the only one whose neck LEARNS the
           upsample.  Its backbone carries no C-block either.
  * v8     ``C2f`` -- v5's C3 with the split moved ahead of the bottlenecks, so
           the concat takes n+2 inputs rather than 2.
  * v9     ``RepNCSPELAN4`` / ``ELAN1`` / ``SPPELAN`` / ``AConv``: GELAN's
           aggregation, a different graph shape from CSP -- branches chained
           and then all concatenated -- plus ``RepConv``.
  * v10    NMS-free ``v10Detect``: dual one2many/one2one heads, so the model
           returns topk-selected boxes rather than a dense grid.  Brings
           ``PSA`` (the first attention block in this file), ``C2fCIB``,
           ``SCDown`` and ``RepVGGDW``.
  * 11     ``C3k2`` (a C2f whose bottleneck is switchable to a nested C3) plus
           ``C2PSA``, position-sensitive attention on the backbone's last
           stage.
  * 12     ``A2C2f`` / ``AAttn`` -- area attention, which splits the feature map
           into regions and attends inside each -- and no SPPF at all.  The most
           attention-heavy preset here.
  * 26     C3k2 + C2PSA like 11, but end2end AND without the DFL: ``reg_max=1``
           and ``self.dfl = Identity``.  It is the pair to v10 that makes
           "end2end" a tested difference rather than an assumed one, and the
           only preset whose box branch regresses distances directly.

``_assert_character`` cashes each claim above against the parsed model.  Without
it a preset can pass while proving nothing: every YAML in this list produces a
model that runs, so "yolo12 passes" is worth nothing if the parser quietly gave
back a stack of plain convolutions.

SCALE.  Every preset runs the smallest published scale (``n``, or ``-tiny`` for
v3) at 64x64 input and ``nc=4``.  Width is the axis that stays honest -- the
nano scales are shipped shapes, not invented ones -- and the two scaled axes are
input resolution and class count, which change the size of the tensors and not
which kernels run.  64 is the smallest input the head accepts: stride 32 has to
leave a feature map with a nonzero extent, and 64 gives 8x8 / 4x4 / 2x2.

    source /workspace/tnpu-env.sh
    python tests/models/Yolo/test_yolo.py --version v8

MEASURED 2026-08-13, one cleared dump path each, nine runs sharing the machine
so the wall clocks are upper bounds.  "kern" is unique compiled kernels; "max
rel" is the worst max|npu-cpu| / max|cpu| over all of a version's outputs.

    version  yaml            params      kern   time   max rel     verdict
    v3       yolov3-tiny  12,134,184      50    384s   1.27e+00    WRONG NUMBERS
    v5       yolov5n       2,509,244      88    746s   1.85e-05    PASS
    v6       yolov6n       4,238,540      68*   420s   --          BLOCKED, device
    v8       yolov8n       3,011,628      89    707s   1.15e-05    PASS
    v9       yolov9t       2,006,188     118   1652s   2.47e+00    WRONG NUMBERS
    v10      yolov10n      2,708,600     124   1024s   1.35e+00    WRONG NUMBERS**
    11       yolo11n       2,590,620     124    907s   5.67e-01    WRONG NUMBERS**
    12       yolo12n       2,568,828     145   1252s   5.83e-01    WRONG NUMBERS**
    26       yolo26n       2,505,360     142   1169s   1.07e+00    WRONG NUMBERS**

     * v6 compiles 68 kernels and then stops; it never reaches an output.
    ** these four SEGFAULTED on the triton-npu pin this file was written
       against.  The pin has since moved and they now RUN and are merely
       wrong; see below.

SO TWO VERSIONS PASS, and the file gates those two.  The other seven each have a
measured stopping point rather than a guess, and no two of them are the same
stop:

  * v6 stops in the DEVICE, not in the compiler.  Its neck upsamples with
    ConvTranspose2d, Inductor does not lower a transposed convolution and emits
    ``extern_kernels.convolution(..., transposed=True, ...)``, and that
    dispatches on npu:0 to ``convolution_overrideable`` -- which
    PyTorchSimDevice/csrc does not register at all.  This is the SAME device gap
    RecurrentGemma's depthwise conv1d hits, and v6 is the evidence that it is
    not a 1-D gap as that note said: the discriminator is whether INDUCTOR
    lowers the convolution, not its rank.  ResNet, MobileNet-v2 and YOLOv5 all
    convolve and pass because theirs are lowered.

  * v3 compiles and runs and the ANSWER IS WRONG, first at the fused
    max_pool2d.  Per-kernel functional verify names it exactly:
    ``triton_npu_fused_max_pool2d_with_indices_silu_3`` writing buf4
    (1, 16, 32, 32) -- the first pool in the model -- with 12167 of 16384
    elements over tolerance and a max abs diff of 4.85 on data of order 1.
    That is a miscompile, not precision.  It does NOT reduce: silu-then-pool at
    the same shape and stride, and the whole Conv-BN-SiLU-MaxPool block
    standalone, both come back clean (9.5e-07), so it needs the model's context.
    Unchanged on triton-npu develop 3434608, on a616ea5 and on develop-refactor
    3c1ccca -- three trees, same 81.83 max diff, so no pin move fixes it.

  * v9 diverges by BRANCH, which is what makes it interesting: its P3 output is
    clean at 9.6e-05 while P4 and P5 come back at 1.25 and 2.47 relative.  A
    uniform precision story cannot produce that.

  * v10 / 11 / 12 / 26 all carry attention (PSA, C2PSA, A2C2f) and all four die
    the same way on the pinned toolchain: a Spike ``User load segfault @
    0x11005bd0`` inside the PSA block's depthwise 3x3 positional-encoding conv
    on the 2x2 feature map.  Bisecting the GRID rather than the data is what
    identified it -- program id 1 alone runs clean, ids 1 and 2 together fault,
    so it is the SECOND invocation that dies, which is a re-entrancy bug and not
    an index overrun.  Bisecting triton-npu over the 23 commits above the pin
    lands on a616ea5, "Take out three broadcast repairs that never fire, and the
    loop one of them held open" (tnpu/passes/p11_select_lane_axis.py).

    THAT COMMIT'S PREMISE DOES NOT HOLD FOR THIS SHAPE, and this kernel is the
    coverage it says cannot be written.  It measured
    ``_replicate_broadcast_on_lane_axis`` at 0 True over 423 calls across a
    352-kernel suite and removed it as dead.  Instrumented at the pinned commit,
    this kernel drives it to True TWICE, and the layout it then chooses is what
    faults.  So the removal is a fix, not a cleanup.

    With a616ea5 all four compile and run -- and then return wrong numbers, so
    the fix is necessary and not sufficient.  yolo11's first divergent buffer is
    buf132, an ``aten.bmm`` of shape (2, 4, 4) inside the attention, 1 element
    of 32 over tolerance at 5.5e-04 relative; by the outputs that has become
    0.57.

    THE PIN NOW CARRIES IT.  ``thirdparty/triton-npu.json`` moved from 2988424
    to 98744f7, which is triton-npu develop MERGED with the p09 unit-axis line
    2988424 sat on.  The merge was needed rather than a bump: 2988424 is not an
    ancestor of develop -- it carries five commits that never landed there --
    so pinning develop alone would have taken the p11 fix and dropped the p09
    fix that test_mllama.py is gated on.  One commit now carries all three:
    p09 (mllama), p11 (this segfault) and p13 (7aec83d, the replicate axes
    Qwen2-VL needs), the last of which landed on develop WHILE this pin was
    being moved and would otherwise have been left behind.

    Re-measured on it: no segfault, and each of the four returns a max rel
    IDENTICAL to its a616ea5 run to every digit (5.67e-01, 5.83e-01, 1.35e+00,
    1.07e+00); test_mllama.py still passes both towers at 4.2915e-06 and
    3.4019e-06; the gate is unchanged at 1.85e-05 and 1.15e-05.

WHAT WOULD HAVE HIDDEN ALL OF THIS.  Every one of these seven passed an earlier
version of this file, and the two reasons are in ``_revive`` and in the
tolerance:  with default weights the activations decay to 1e-09 by P5, and with
a flat atol=1e-2 every comparison against them is vacuous.  v3 went from "max
diff 3.2e-03, PASSED" to a 127% error on the same build, with only the weights
and the tolerance changed.  A green YOLO run is worth exactly what its
activations and its tolerance are worth.
"""

import argparse
import copy
import logging
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


# One preset per parser version directory under ultralytics/cfg/models.  The
# YAML name is what selects the scale: parse_model reads it back out of the
# stem (yolov8n -> scale "n"), which is the same path a user takes.
_PRESETS = {
    "v3": dict(yaml="yolov3-tiny.yaml", imgsz=64, nc=4),
    "v5": dict(yaml="yolov5n.yaml", imgsz=64, nc=4),
    "v6": dict(yaml="yolov6n.yaml", imgsz=64, nc=4),
    "v8": dict(yaml="yolov8n.yaml", imgsz=64, nc=4),
    "v9": dict(yaml="yolov9t.yaml", imgsz=64, nc=4),
    "v10": dict(yaml="yolov10n.yaml", imgsz=64, nc=4),
    "11": dict(yaml="yolo11n.yaml", imgsz=64, nc=4),
    "12": dict(yaml="yolo12n.yaml", imgsz=64, nc=4),
    "26": dict(yaml="yolo26n.yaml", imgsz=64, nc=4),
}

# What the parser MUST have instantiated for the preset to mean what it claims,
# and what it must NOT have -- the second half is what keeps a preset from
# silently becoming a copy of its neighbour.
_CHARACTER = {
    "v3":  dict(needs=("ZeroPad2d", "MaxPool2d"),
                forbids=("C3", "C2f", "C3k2", "A2C2f", "SPPF", "Bottleneck")),
    "v5":  dict(needs=("C3", "Bottleneck", "SPPF"),
                forbids=("C2f", "C3k2", "A2C2f", "ConvTranspose2d")),
    "v6":  dict(needs=("ConvTranspose2d", "SPPF"),
                forbids=("C3", "C2f", "C3k2", "A2C2f", "Upsample")),
    "v8":  dict(needs=("C2f", "Bottleneck", "SPPF"),
                forbids=("C3", "C3k2", "C2PSA", "A2C2f", "ConvTranspose2d")),
    "v9":  dict(needs=("RepNCSPELAN4", "ELAN1", "SPPELAN", "AConv", "RepConv"),
                forbids=("C3", "C2f", "C3k2", "SPPF", "A2C2f")),
    "v10": dict(needs=("C2fCIB", "PSA", "SCDown", "RepVGGDW", "v10Detect"),
                forbids=("C3k2", "A2C2f", "C2PSA")),
    "11":  dict(needs=("C3k2", "C2PSA", "PSABlock", "SPPF"),
                forbids=("C2f", "A2C2f", "v10Detect")),
    "12":  dict(needs=("A2C2f", "AAttn", "C3k2"),
                forbids=("SPPF", "C2PSA", "v10Detect", "MaxPool2d")),
    "26":  dict(needs=("C3k2", "C2PSA", "SPPF"),
                forbids=("C2f", "A2C2f", "v10Detect")),
}

# Every version this parser builds gets the anchor-free head; 26 is the one that
# drops the DFL with it (reg_max 1, Identity in place of the 16-bin softmax).
_NO_DFL = {"26"}
# NMS-free: the model itself returns selected boxes, not a dense grid.
_END2END = {"v10", "26"}


def _quiet_ultralytics():
    """Ultralytics logs a full layer table per build; the test prints its own."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)


def _revive(model, imgsz, batch=8):
    """Give every BatchNorm the statistics of its own input, in one pass.

    THIS IS NOT COSMETIC, and the test proves much less without it.  An
    untrained BatchNorm in eval mode is a PASSTHROUGH -- running_mean 0,
    running_var 1 -- so nothing renormalises between layers, and PyTorch's
    default Conv2d init shrinks the activation variance at every one of them.
    Measured on v8: the three neck outputs come out at 3.0e-05, 2.7e-07 and
    4.5e-09.  At that size the comparison is decided by the detection head's
    bias and by the anchor grid -- both constants this backend barely touches
    -- and a backbone computing garbage would still pass.

    Calibrating is what a BatchNorm is FOR, and it is one forward pass: a
    pre-hook writes each BN's running stats from the activation about to enter
    it, so by the time the pass reaches layer N every layer before it has
    already been normalised.  v8's three neck outputs afterwards: 10, 14, 17.
    Every version lands between 4 and 1200 across all outputs, which is what
    makes ONE relative tolerance meaningful for all of them.

    A GAIN-ONLY FIX WAS TRIED FIRST AND IS WORSE.  Re-initialising the convs
    Kaiming fan-in/relu also clears the decay, but nothing bounds the result
    afterwards: v12 came out at 1.6e+05 and yolov3-tiny diverged outright --
    its DFL softmax saturated on the device to bin 15, so every decoded box
    was anchor + 15*16 = 240 where the CPU had a spread.  Normalising per
    layer is what keeps the dynamic range in a range both sides agree on.

    Transformer tests in this suite need no such thing: a LayerNorm or RMSNorm
    renormalises every block, so their random-weight activations stay O(1).
    """
    import torch.nn as nn

    hooks = []

    def _stats(mod, inputs):
        a = inputs[0].detach()
        dims = [0] + list(range(2, a.dim()))
        mod.running_mean.copy_(a.mean(dim=dims))
        # clamp_min: a channel that is constant across the calibration batch
        # would otherwise divide by zero and take the whole model to inf.
        mod.running_var.copy_(a.var(dim=dims, unbiased=False).clamp_min(1e-5))

    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            hooks.append(mod.register_forward_pre_hook(_stats))

    # Its own generator, so the calibration input is not the test input.
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model(torch.randn(batch, 3, imgsz, imgsz, generator=g))
    for h in hooks:
        h.remove()
    return model


def _build(version, imgsz=None):
    """Parse the version's YAML into a model.  No checkpoint, no network."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    p = _PRESETS[version]
    # Calibrate at the resolution the test will actually run: BN statistics
    # taken at 64 do not describe the activations at 128.
    imgsz = imgsz if imgsz is not None else p["imgsz"]
    cfg = yaml_model_load(p["yaml"])
    # Seed before construction: the weights are random, so without this the
    # worst element wanders across the threshold and the test is flaky.
    torch.manual_seed(0)
    model = DetectionModel(cfg=cfg, ch=3, nc=p["nc"], verbose=False).eval()
    return _revive(model, imgsz), cfg


def _module_names(model):
    """Every module class name in the parsed graph, nested blocks included."""
    return {type(m).__name__ for m in model.modules()}


def _assert_character(version, model, cfg, imgsz=None):
    """Fail if the parser did not produce the blocks this preset is named for."""
    names = _module_names(model)
    spec = _CHARACTER[version]

    for want in spec["needs"]:
        assert want in names, (
            f"preset {version} claims {want}, which the parser did not instantiate; "
            f"this model is not the {version} architecture"
        )
    for forbid in spec["forbids"]:
        assert forbid not in names, (
            f"preset {version} must not contain {forbid} -- it would make this "
            f"preset a copy of another one"
        )

    head = model.model[-1]
    has_dfl = type(getattr(head, "dfl", None)).__name__ == "DFL"
    if version in _NO_DFL:
        assert not has_dfl and head.reg_max == 1, (
            f"{version} drops the DFL (reg_max 1, Identity); this head has "
            f"reg_max={head.reg_max} and a {type(head.dfl).__name__}"
        )
    else:
        assert has_dfl and head.reg_max == 16, (
            f"{version} runs the 16-bin DFL head through this parser; got "
            f"reg_max={getattr(head, 'reg_max', None)}"
        )

    end2end = bool(getattr(head, "end2end", False))
    assert end2end == (version in _END2END), (
        f"{version}: end2end is {end2end}; the preset exists for the opposite"
    )

    # A scaled-down input still has to leave the deepest stride a real feature
    # map -- otherwise the neck's concats degenerate and the run proves nothing.
    imgsz = imgsz if imgsz is not None else _PRESETS[version]["imgsz"]
    max_stride = int(max(model.stride))
    assert imgsz % max_stride == 0 and imgsz // max_stride >= 2, (
        f"imgsz {imgsz} against max stride {max_stride} leaves a "
        f"{imgsz // max_stride}x{imgsz // max_stride} map; too small to be a test"
    )


def _tensors(obj, out=None):
    """Flatten every tensor in a YOLO output, whatever the version wrapped it in.

    v3/v5/v8/v9/11 return (decoded, {boxes, scores, feats}); v10 and 26 return
    (selected, {one2many, one2one}).  Comparing the flattened list rather than
    just element 0 keeps the raw feature maps in the comparison, which is where
    the backbone and neck actually show up.
    """
    out = [] if out is None else out
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            _tensors(obj[k], out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _tensors(v, out)
    return out


@torch.no_grad()
def run_yolo(device, version="v8", batch=1, imgsz=None, compile_model=True, rtol=1e-4, atol=1e-6):
    _quiet_ultralytics()
    p = _PRESETS[version]
    imgsz = imgsz if imgsz is not None else p["imgsz"]

    model_cpu, cfg = _build(version, imgsz)
    n_params = sum(x.numel() for x in model_cpu.parameters())
    top = [type(m).__name__ for m in model_cpu.model]
    print(f"version={version} yaml={p['yaml']} scale={cfg.get('scale')} "
          f"imgsz={imgsz} nc={p['nc']} batch={batch}")
    print(f"  params={n_params} layers={len(model_cpu.model)} "
          f"stride={[int(s) for s in model_cpu.stride]}")
    print(f"  blocks: {', '.join(sorted(set(top)))}")

    _assert_character(version, model_cpu, cfg, imgsz)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, 3, imgsz, imgsz, generator=g)

    cpu_out = _tensors(model_cpu(x))

    model_npu = copy.deepcopy(model_cpu).to(device).eval()
    if compile_model:
        model_npu = torch.compile(model_npu, dynamic=False)
    npu_out = _tensors(model_npu(x.to(device)))

    assert len(npu_out) == len(cpu_out), (
        f"output structure differs: {len(npu_out)} tensors on npu, {len(cpu_out)} on cpu"
    )
    # THE TOLERANCE IS PER OUTPUT, and a single atol would make most of these
    # comparisons vacuous.  These outputs do not share a scale: the decoded
    # boxes are pixel coordinates and run to 480, while a P5 neck map is O(0.1).
    # A flat atol=1e-2 would pass ANY value at all for the second -- every
    # element is already within 1e-2 of zero -- so each output gets
    # atol = tol * max|ref| instead, floored so an all-zero tensor still has a
    # threshold.  That is one criterion, applied at nine different scales.
    # The whole table is printed BEFORE any assertion.  test_result exits on
    # the first mismatch, and a failure that reports one output while hiding
    # the other eight says nothing about where the divergence starts.
    worst = worst_rel = 0.0
    rows = []
    for i, (a, b) in enumerate(zip(npu_out, cpu_out)):
        scale = torch.max(torch.abs(b)).item()
        scaled_atol = max(atol, rtol * scale)
        diff = torch.max(torch.abs(a.cpu() - b)).item()
        rel = diff / scale if scale else 0.0
        verdict = "ok" if diff <= scaled_atol + rtol * scale else "DIVERGED"
        print(f"  output[{i}] {str(tuple(b.shape)):>20}  max|d|={diff:.4e}  "
              f"max|ref|={scale:.4e}  rel={rel:.2e}  atol={scaled_atol:.2e}  {verdict}")
        rows.append((i, a, b, scaled_atol))
        worst, worst_rel = max(worst, diff), max(worst_rel, rel)
    print(f"Max diff >  {worst}")
    print(f"Max rel  >  {worst_rel}")
    for i, a, b, scaled_atol in rows:
        test_result(f"YOLO {version} output[{i}] {tuple(b.shape)}", a, b,
                    rtol=rtol, atol=scaled_atol)
    print("Yolo Simulation Done")


# THE GATE: what the sweep runs when it runs this file with no arguments.
#
# The two versions that pass, and they are not redundant: v5 is C3 + SPPF with
# the anchor-era backbone, v8 is C2f, and between them they cover both CSP
# shapes plus the DFL head every later version inherits.  MEASURED AS THE SWEEP
# RUNS IT -- one process, no arguments, 1128s -- inside the sweep's 1800s.  The
# two runs share codegen rather than colliding in it: 1453s if run separately,
# and each version's max rel is identical either way (1.85e-05 and 1.15e-05).
# THAT NUMBER IS LOAD-SENSITIVE and the margin is not large: the same gate run
# took 1578s on a machine also building a toolchain and running eight other
# models.  Same two diffs, to every digit, on the moved pin.
#
# The other seven are one --version each and are NOT gated; each has a measured
# stopping point in the module docstring.  A version is added here when it
# passes on the pinned toolchain, not when it merely compiles.
_GATE = ("v5", "v8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO, one preset per parser version")
    parser.add_argument("--version", type=str, default=None, choices=list(_PRESETS),
                        help=f"one version; the default runs the gate ({', '.join(_GATE)})")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="relative criterion; also scales each output's atol")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="absolute floor, for an output whose reference is all zero")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    torch.compiler.is_compiling = lambda: True  # FIXME. Same as test_llama.py.

    for version in ([args.version] if args.version else _GATE):
        run_yolo(
            torch.device("npu:0"),
            version=version,
            batch=args.batch,
            imgsz=args.imgsz,
            compile_model=args.compile,
            rtol=args.rtol,
            atol=args.atol,
        )
