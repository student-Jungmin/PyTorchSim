"""Write timing_cases.csv from the per-operation case files.

The case files are the source of truth; this adds the derived columns (MACs,
DRAM bytes at both widths, the roofline side per machine) and rewrites PARAMS.md.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cases import load  # noqa: E402
from roofline import MACHINES, cost, regime  # noqa: E402

OUT = os.path.join(HERE, "timing_cases.csv")


def main():
    rows = []
    for c in load():
        macs, byts = cost(c["op"], c["size"])
        half = byts // 2
        rows.append({
            "workload": c["name"], "op": c["op"], "size": " ".join(map(str, c["size"])),
            "origin": c["origin"], "census_match": c["census"],
            "params": c["params"], "macs": macs, "bytes_fp32": byts, "bytes_fp16": half,
            "intensity_fp32": round(macs / byts, 2) if byts else 0.0,
            "intensity_fp16": round(macs / half, 2) if half else 0.0,
            "regime_v3_fp32": regime(macs, byts, "v3"),
            "regime_v3_fp16": regime(macs, half, "v3"),
            "regime_v6e_fp32": regime(macs, byts, "v6e"),
            "regime_v6e_fp16": regime(macs, half, "v6e"),
            "source": c["source"], "note": c["note"]})

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import collections
    print(f"{len(rows)} cases -> {OUT}")
    print("origin  ", dict(collections.Counter(r["origin"] for r in rows)))
    for m in MACHINES:
        for d in ("fp32", "fp16"):
            print(f"{m:4s} {d}", dict(collections.Counter(r[f"regime_{m}_{d}"] for r in rows)))
    print({k: round(v["ridge"], 1) for k, v in MACHINES.items()}, "MAC/byte")

    import params_md
    params_md.main()


if __name__ == "__main__":
    main()
