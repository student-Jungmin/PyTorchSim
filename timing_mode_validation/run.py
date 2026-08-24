"""Run timing-mode cases: one process per case, one log per (case, dtype).

`python run.py --op gemm` or `python ops/gemm.py --origin census`.
Timing mode is forced on here because this branch defaults it off.
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cases import load, ops  # noqa: E402

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", os.path.dirname(HERE))
LOG_DIR = os.path.join(HERE, "logs")


def log_path(name, dtype):
    """Where one case's log goes; the width is in the name unless it is fp32."""
    suffix = "" if dtype == "float32" else f".{dtype}"
    return os.path.join(LOG_DIR, f"{name}{suffix}.log")


def run_case(case, dtype, isolate):
    """One case in its own process, tee'd to its log. Returns (name, ok)."""
    env = dict(os.environ)
    env["TORCHSIM_TIMING_MODE"] = "1"
    v6e = os.path.join(TORCHSIM_DIR,
                       "configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_timing_only.yml")
    v3 = os.path.join(TORCHSIM_DIR,
                      "configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml")
    env.setdefault("TOGSIM_CONFIG", v6e if os.path.exists(v6e) else v3)
    if isolate:
        root = env.get("TORCHSIM_DUMP_PATH_ROOT", os.path.join(TORCHSIM_DIR, "outputs"))
        env["TORCHSIM_DUMP_PATH"] = os.path.join(root, f"case_{case['name']}")
        os.makedirs(env["TORCHSIM_DUMP_PATH"], exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "bench.py"), "--op", case["op"],
           "--dtype", dtype, "--size"] + [str(d) for d in case["size"]]
    with open(log_path(case["name"], dtype), "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return case["name"], proc.returncode == 0


def main(op_file=None):
    """Entry point for both `run.py --op X` and `ops/X.py`."""
    ap = argparse.ArgumentParser(description="Run timing-mode validation cases")
    if op_file is None:
        ap.add_argument("--op", default="", help="comma-separated ops; default all")
    ap.add_argument("--origin", default="", help="census,sweep,anchor,added; default all")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--jobs", type=int, default=1, help="cases in parallel")
    ap.add_argument("--isolate", action="store_true", help="a dump path per case")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if op_file is not None:
        targets = [op_file]
    else:
        targets = [o for o in args.op.split(",") if o] or ops()

    selected = load(targets, origin=set(o for o in args.origin.split(",") if o))
    print(f"[*] {len(selected)} cases  dtype={args.dtype} origin={args.origin or 'all'}")
    if args.list:
        for c in selected:
            print(f"  {c['origin']:<6} {c['census']:<5} {c['name']:<34} "
                  f"{c['op']} {' '.join(map(str, c['size']))}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, c, args.dtype, args.isolate) for c in selected]
        for fut in concurrent.futures.as_completed(futures):
            name, ok = fut.result()
            done += 1
            print(f"[{done}/{len(selected)}] {'ok  ' if ok else 'FAIL'} {name}", flush=True)


if __name__ == "__main__":
    main()
