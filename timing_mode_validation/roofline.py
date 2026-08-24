"""MACs, DRAM bytes and the roofline side each case falls on, per machine.

The two machines are read off the configs in `configs/`, not guessed: a case's
regime is whichever of the array and the DRAM would take longer.
"""

#: MAC/byte at which the array and DRAM cost the same, per machine.
#:   v3  = systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml
#:         2 x 128x128 @ 940 MHz = 32,768 MAC/cycle; HBM2 ~450 GB/s per core
#:   v6e = systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
#:         2 x 256x256 @ 3502 MHz = 131,072 MAC/cycle (918 TFLOP/s, the shipped
#:         number); HBM3 32 ch x 2 pseudo x 32 bit x 6400 MT/s = 1638 GB/s
MACHINES = {
    "v3": {"mac_per_cycle": 2 * 128 * 128, "clock_hz": 940e6, "dram_bytes_s": 450e9},
    "v6e": {"mac_per_cycle": 2 * 256 * 256, "clock_hz": 3502e6, "dram_bytes_s": 1638.4e9},
}

for _m in MACHINES.values():
    _m["bytes_per_cycle"] = _m["dram_bytes_s"] / _m["clock_hz"]
    _m["ridge"] = _m["mac_per_cycle"] / _m["bytes_per_cycle"]


def regime(macs, byts, machine):
    """MXU / DRAM / VPU-DRAM for one case at one element width."""
    if macs == 0:
        return "VPU/DRAM"
    return "MXU" if macs / byts >= MACHINES[machine]["ridge"] else "DRAM"


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def cost(op, s):
    """(MAC count, fp32 DRAM bytes) for one case, from its shape alone."""
    if op in ("gemm", "gemm_bias"):
        m, k, n = s
        return m * k * n, 4 * (m * k + k * n + m * n)
    if op == "rope_outer":
        r, d = s
        return r * d, 4 * (r + d + r * d)
    if op == "bmm":
        b, m, k, n = s
        return b * m * k * n, 4 * b * (m * k + k * n + m * n)
    if op == "attn_qk":
        b, ss, d = s
        return b * ss * ss * d, 4 * b * (2 * ss * d + ss * ss)
    if op == "attn_pv":
        b, ss, d = s
        return b * ss * ss * d, 4 * b * (ss * ss + 2 * ss * d)
    if op in ("attention", "attn_causal"):
        b, ss, d = s
        return 2 * b * ss * ss * d, 4 * b * (3 * ss * d + ss * ss)
    if op == "attn_decode":
        b, kv, d = s
        return 2 * b * kv * d, 4 * b * (2 * kv * d + d + kv)
    if op == "gqa_attn":
        hq, hkv, ss, d = s
        return 2 * hq * ss * ss * d, 4 * (hq * ss * d + 2 * hkv * ss * d + hq * ss * ss)
    if op in ("softmax", "layernorm", "swiglu", "residual", "reduce_sum", "rmsnorm", "gelu"):
        r, c = s
        return 0, 4 * r * c * (3 if op in ("swiglu", "residual") else 2)
    if op == "softmax3d":
        b, r, c = s
        return 0, 4 * 2 * b * r * c
    if op == "rope":
        b, h, ss, d = s
        return 0, 4 * (2 * b * h * ss * d + 2 * ss * d)
    if op == "embedding":
        _v, h, t = s
        return 0, 4 * (t * h * 2)
    if op in ("topk", "moe_route"):
        t, e, k = s
        return 0, 4 * (t * e + t * k * 2)
    if op == "dispatch":
        t, _k, h = s
        return 0, 4 * (t * h * 3)
    if op == "sort1d":
        t, e, k = s
        return 0, 4 * (t * e + 2 * t * k)
    if op == "conv":
        n, h, w, c, k, r, st, _pad = s
        oh, ow = h // st, w // st
        return n * oh * ow * k * c * r * r, 4 * (n * c * h * w + k * c * r * r + n * k * oh * ow)
    if op == "dwconv":
        n, c, h, w, r, st = s
        oh, ow = h // st, w // st
        return n * oh * ow * c * r * r, 4 * (n * c * h * w + c * r * r + n * c * oh * ow)
    if op == "pwconv":
        n, c, h, w, k = s
        return n * h * w * k * c, 4 * (n * c * h * w + k * c + n * k * h * w)
    if op == "patch_embed":
        n, c, h, k, p = s
        g = h // p
        return n * g * g * k * c * p * p, 4 * (n * c * h * h + k * c * p * p + n * k * g * g)
    if op == "conv1d_causal":
        n, c, l, r = s
        return n * c * l * r, 4 * (2 * n * c * l + c * r)
    if op == "convtranspose":
        # The MACs are the ones the backend EMITS, not the transposed
        # convolution's own. inductor_templates rewrites it as the direct
        # convolution over a zero-spread input, so the work is the OUTPUT
        # grid's -- st^2 times the ideal, with those extra products against
        # inserted zeros.
        n, c, h, w, k, r, st = s
        oh, ow = (h - 1) * st + r, (w - 1) * st + r
        return (n * oh * ow * c * k * r * r,
                4 * (n * c * h * w + c * k * r * r + n * k * oh * ow))
    if op == "maxpool":
        n, c, h, w, k, st = s
        return 0, 4 * (n * c * h * w + n * c * (h // st) * (w // st))
    if op == "mlp_swiglu":
        ss, hid, it = s
        return 3 * ss * hid * it, 4 * (ss * hid + 3 * hid * it + 2 * ss * it)
    if op == "attn_block":
        ss, hid, hq, hkv = s
        d = hid // hq
        proj = 2 * ss * hid * hid + 2 * ss * hid * hkv * d
        return proj + 2 * hq * ss * ss * d, 4 * (2 * ss * hid + 2 * hid * hid + 2 * hid * hkv * d)
    raise KeyError(f"no cost model for op {op}")
