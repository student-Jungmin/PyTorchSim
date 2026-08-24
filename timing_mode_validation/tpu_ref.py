"""Measure the same cases on a real TPU and write workload,cycles for the join.

Runs on a TPU host with JAX installed; it is the reference side of
``timing_cases.csv`` and never touches PyTorchSim.  NOT RUN IN THIS REPO'S CI --
it needs hardware, so treat the numbers it writes as measurements to review.
"""

import argparse
import csv
import math
import os
import statistics
import time

import jax
import jax.numpy as jnp

CLOCK_MHZ = {"v2": 700, "v3": 940, "v4": 1050, "v5e": 1670, "v5p": 1750, "v6e": 1740}


def _rand(key, *shape, dtype):
    return jax.random.normal(key, shape, dtype=dtype)


def build(op, size, dtype, prec):
    """The JAX twin of one op_bench case: a jitted fn and its device operands."""
    k = jax.random.PRNGKey(0)
    ks = jax.random.split(k, 8)
    mm = lambda a, b: jnp.matmul(a, b, precision=prec)

    if op in ("gemm", "gemm_bias"):
        M, K, N = size
        a, b = _rand(ks[0], M, K, dtype=dtype), _rand(ks[1], K, N, dtype=dtype)
        if op == "gemm":
            return mm, (a, b)
        c = _rand(ks[2], N, dtype=dtype)
        return (lambda a, b, c: mm(a, b) + c), (a, b, c)
    if op == "rope_outer":
        R, D = size
        return mm, (_rand(ks[0], R, 1, dtype=dtype), _rand(ks[1], 1, D, dtype=dtype))
    if op == "bmm":
        B, M, K, N = size
        return mm, (_rand(ks[0], B, M, K, dtype=dtype), _rand(ks[1], B, K, N, dtype=dtype))
    if op == "attn_qk":
        B, S, D = size
        return ((lambda q, kk: mm(q, jnp.swapaxes(kk, -1, -2))),
                (_rand(ks[0], B, S, D, dtype=dtype), _rand(ks[1], B, S, D, dtype=dtype)))
    if op == "attn_pv":
        B, S, D = size
        return mm, (_rand(ks[0], B, S, S, dtype=dtype), _rand(ks[1], B, S, D, dtype=dtype))
    if op in ("attention", "attn_causal"):
        B, S, D = size
        def fn(q, kk, v, *rest):
            s = mm(q, jnp.swapaxes(kk, -1, -2)) / math.sqrt(D)
            if rest:
                s = s + rest[0]
            return mm(jax.nn.softmax(s, axis=-1), v)
        ops = tuple(_rand(ks[i], B, S, D, dtype=dtype) for i in range(3))
        if op == "attn_causal":
            mask = jnp.triu(jnp.full((S, S), -jnp.inf, dtype=dtype), 1)
            ops = ops + (mask,)
        return fn, ops
    if op == "attn_decode":
        B, KV, D = size
        def fn(q, kk, v):
            s = mm(q, jnp.swapaxes(kk, -1, -2)) / math.sqrt(D)
            return mm(jax.nn.softmax(s, axis=-1), v)
        return fn, (_rand(ks[0], B, 1, D, dtype=dtype), _rand(ks[1], B, KV, D, dtype=dtype),
                    _rand(ks[2], B, KV, D, dtype=dtype))
    if op == "gqa_attn":
        Hq, Hkv, S, D = size
        rep = Hq // Hkv
        def fn(q, kk, v):
            k2 = jnp.repeat(kk, rep, axis=0)
            v2 = jnp.repeat(v, rep, axis=0)
            s = mm(q, jnp.swapaxes(k2, -1, -2)) / math.sqrt(D)
            return mm(jax.nn.softmax(s, axis=-1), v2)
        return fn, (_rand(ks[0], Hq, S, D, dtype=dtype), _rand(ks[1], Hkv, S, D, dtype=dtype),
                    _rand(ks[2], Hkv, S, D, dtype=dtype))
    if op == "softmax":
        R, C = size
        return (lambda x: jax.nn.softmax(x, axis=-1)), (_rand(ks[0], R, C, dtype=dtype),)
    if op == "softmax3d":
        B, R, C = size
        return (lambda x: jax.nn.softmax(x, axis=-1)), (_rand(ks[0], B, R, C, dtype=dtype),)
    if op == "rmsnorm":
        R, C = size
        def fn(x, w):
            return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * w
        return fn, (_rand(ks[0], R, C, dtype=dtype), _rand(ks[1], C, dtype=dtype))
    if op == "layernorm":
        R, C = size
        def fn(x, w, b):
            m = jnp.mean(x, -1, keepdims=True)
            v = jnp.mean((x - m) ** 2, -1, keepdims=True)
            return (x - m) * jax.lax.rsqrt(v + 1e-5) * w + b
        return fn, (_rand(ks[0], R, C, dtype=dtype), _rand(ks[1], C, dtype=dtype),
                    _rand(ks[2], C, dtype=dtype))
    if op == "swiglu":
        R, C = size
        return ((lambda a, b: jax.nn.silu(a) * b),
                (_rand(ks[0], R, C, dtype=dtype), _rand(ks[1], R, C, dtype=dtype)))
    if op == "gelu":
        R, C = size
        return (lambda x: jax.nn.gelu(x, approximate=True)), (_rand(ks[0], R, C, dtype=dtype),)
    if op == "residual":
        R, C = size
        return ((lambda a, b: a + b),
                (_rand(ks[0], R, C, dtype=dtype), _rand(ks[1], R, C, dtype=dtype)))
    if op == "reduce_sum":
        R, C = size
        return (lambda x: jnp.sum(x, -1)), (_rand(ks[0], R, C, dtype=dtype),)
    if op == "rope":
        B, H, S, D = size
        def fn(x, cos, sin):
            x1, x2 = x[..., : D // 2], x[..., D // 2:]
            return x * cos + jnp.concatenate([-x2, x1], axis=-1) * sin
        return fn, (_rand(ks[0], B, H, S, D, dtype=dtype), _rand(ks[1], 1, 1, S, D, dtype=dtype),
                    _rand(ks[2], 1, 1, S, D, dtype=dtype))
    if op == "embedding":
        V, H, T = size
        ids = jax.random.randint(ks[1], (1, T), 0, V)
        return (lambda w, i: jnp.take(w, i, axis=0)), (_rand(ks[0], V, H, dtype=dtype), ids)
    if op == "topk":
        T, E, K = size
        return (lambda x: jax.lax.top_k(x, K)), (_rand(ks[0], T, E, dtype=dtype),)
    if op == "moe_route":
        T, E, K = size
        def fn(x):
            w = jax.nn.softmax(x, axis=-1)
            v, i = jax.lax.top_k(w, K)
            return v / jnp.sum(v, -1, keepdims=True), i
        return fn, (_rand(ks[0], T, E, dtype=dtype),)
    if op == "dispatch":
        T, K, H = size
        idx = jax.random.randint(ks[1], (T * K,), 0, T)
        def fn(x, i):
            picked = jnp.take(x, i, axis=0)
            return jnp.zeros_like(x).at[i].add(picked)
        return fn, (_rand(ks[0], T, H, dtype=dtype), idx)
    if op == "sort1d":
        (N,) = size
        return (lambda x: jnp.sort(x)), (jax.random.randint(ks[0], (N,), 0, 256),)
    if op == "conv":
        N, H, W, C, K, R, st, pad = size
        return ((lambda x, w: jax.lax.conv_general_dilated(
                    x, w, (st, st), ((pad, pad), (pad, pad)), precision=prec)),
                (_rand(ks[0], N, C, H, W, dtype=dtype), _rand(ks[1], K, C, R, R, dtype=dtype)))
    if op in ("dwconv", "pwconv", "patch_embed"):
        if op == "dwconv":
            N, C, H, W, R, st = size
            x, w, groups, stride, pad = ((N, C, H, W), (C, 1, R, R), C, (st, st),
                                         ((R // 2, R // 2), (R // 2, R // 2)))
        elif op == "pwconv":
            N, C, H, W, K = size
            x, w, groups, stride, pad = (N, C, H, W), (K, C, 1, 1), 1, (1, 1), ((0, 0), (0, 0))
        else:
            N, C, H, K, P = size
            x, w, groups, stride, pad = (N, C, H, H), (K, C, P, P), 1, (P, P), ((0, 0), (0, 0))
        return ((lambda a, b: jax.lax.conv_general_dilated(
                    a, b, stride, pad, feature_group_count=groups, precision=prec)),
                (_rand(ks[0], *x, dtype=dtype), _rand(ks[1], *w, dtype=dtype)))
    if op == "maxpool":
        N, C, H, W, k, st = size
        return ((lambda x: jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                                                 (1, 1, k, k), (1, 1, st, st), "SAME")),
                (_rand(ks[0], N, C, H, W, dtype=dtype),))
    if op == "mlp_swiglu":
        S, hid, it = size
        def fn(x, wg, wu, wd):
            return mm(jax.nn.silu(mm(x, wg)) * mm(x, wu), wd)
        return fn, (_rand(ks[0], S, hid, dtype=dtype), _rand(ks[1], hid, it, dtype=dtype),
                    _rand(ks[2], hid, it, dtype=dtype), _rand(ks[3], it, hid, dtype=dtype))
    if op == "attn_block":
        S, hid, hq, hkv = size
        D = hid // hq
        rep = hq // hkv
        def fn(x, wq, wk, wv, wo):
            q = jnp.swapaxes(mm(x, wq).reshape(S, hq, D), 0, 1)
            kk = jnp.swapaxes(mm(x, wk).reshape(S, hkv, D), 0, 1)
            v = jnp.swapaxes(mm(x, wv).reshape(S, hkv, D), 0, 1)
            kk, v = jnp.repeat(kk, rep, 0), jnp.repeat(v, rep, 0)
            s = mm(q, jnp.swapaxes(kk, -1, -2)) / math.sqrt(D)
            o = mm(jax.nn.softmax(s, -1), v)
            return mm(jnp.swapaxes(o, 0, 1).reshape(S, hid), wo)
        return fn, (_rand(ks[0], S, hid, dtype=dtype), _rand(ks[1], hid, hid, dtype=dtype),
                    _rand(ks[2], hid, hkv * D, dtype=dtype), _rand(ks[3], hid, hkv * D, dtype=dtype),
                    _rand(ks[4], hid, hid, dtype=dtype))
    raise SystemExit(f"no TPU twin for op {op}")


def measure(fn, operands, iters, repeats):
    """Median wall time of one call, with the loop overhead already amortised."""
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*operands))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = jitted(*operands)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "timing_cases.csv"))
    ap.add_argument("--out", default="tpu_ref.csv")
    ap.add_argument("--op", default="", help="comma-separated ops; default all")
    ap.add_argument("--chip", default="v3", choices=sorted(CLOCK_MHZ))
    # WIDTH PAIRING. The simulated device has an fp16 path and refuses bf16
    # (no bf16 in the modelled ISA); the TPU MXU is bf16-native and converts
    # fp16. So the two-byte column is fp16 on our side and bfloat16 here --
    # same traffic, same multiplier class, different mantissa. Say so in reports.
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--precision", default="highest", choices=["default", "high", "highest"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    dtype = getattr(jnp, args.dtype)
    prec = getattr(jax.lax.Precision, args.precision.upper())
    clock = CLOCK_MHZ[args.chip] * 1e6
    ops_only = set(o for o in args.op.split(",") if o)

    rows = []
    with open(args.cases, newline="") as f:
        for case in csv.DictReader(f):
            if ops_only and case["op"] not in ops_only:
                continue
            size = [int(x) for x in case["size"].split()]
            try:
                fn, operands = build(case["op"], size, dtype, prec)
                sec = measure(fn, operands, args.iters, args.repeats)
            except Exception as exc:                      # a shape the chip cannot hold
                print(f"{case['workload']:<40} SKIP {type(exc).__name__}: {exc}")
                continue
            cyc = sec * clock
            rows.append({"workload": case["workload"], "cycles": round(cyc),
                         "seconds": f"{sec:.9f}", "chip": args.chip, "dtype": args.dtype,
                         "precision": args.precision})
            print(f"{case['workload']:<40} {sec * 1e6:9.2f} us  {cyc:12.0f} cycles")

    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows) -- copy to "
          "timing_mode_validation/tpu_ref.csv on the PyTorchSim host")


if __name__ == "__main__":
    main()
