"""Write PARAMS.md: the value range of every parameter, per operation.

Generated from `ops/*.py` so it cannot drift from the cases; one row per
(operation, parameter), with the model-measured values kept apart from the added ones.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cases import load, ops  # noqa: E402


def descriptions():
    """One line per op, read from its builder's docstring in bench.py."""
    import ast
    src = open(os.path.join(HERE, "bench.py")).read()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            if doc:
                out[node.name] = doc
    return out

OUT = os.path.join(HERE, "PARAMS.md")


def _fmt(vals, limit=14):
    """A value list, elided in the middle when it is long."""
    if not vals:
        return "—"
    v = sorted(set(vals))
    if len(v) <= limit:
        return ", ".join(str(x) for x in v)
    head = ", ".join(str(x) for x in v[:limit - 3])
    return f"{head}, … , " + ", ".join(str(x) for x in v[-3:])


def main():
    desc = descriptions()
    body, total = [], 0
    for op in ops():
        cases = load([op])
        total += len(cases)
        names = [n.split("(")[0].strip() for n in cases[0]["params"].split(", ")]
        real = [c for c in cases if c["origin"] != "added"]
        added = [c for c in cases if c["origin"] == "added"]
        body += [f"## `{op}` — {len(cases)}개 "
                  f"(모델 {len(real)} / 추가 {len(added)})", "",
                  desc.get(op, ""), "",
                  "| 파라미터 | 범위 | 모델 값 | 추가 값 |", "|---|---|---|---|"]
        for i, name in enumerate(names):
            col = [c["size"][i] for c in cases]
            lo, hi = min(col), max(col)
            rng = str(lo) if lo == hi else f"{lo} – {hi}"
            body.append(f"| `{name}` | {rng} | {_fmt([c['size'][i] for c in real])} | "
                         f"{_fmt([c['size'][i] for c in added])} |")
        body.append("")

    head = [
        "# 연산별 파라미터 범위", "",
        f"**{len(ops())}개 연산 / {total}개 케이스.** `ops/*.py`에서 생성한다 "
        "(`python timing_mode_validation/index.py`) — 케이스를 고치면 이 파일도 다시 만들어진다.",
        "",
        "- **범위**는 그 파라미터가 도는 최소~최대다. 축이 여러 개인 연산은 모든 조합을 "
        "도는 것이 아니라 케이스가 정한 조합만 돈다 — 조합은 `timing_cases.csv`에 있다.",
        "- **모델 값**은 캡처된 모델(`OP_CENSUS.md`)에서 온 shape가 쓰는 값, "
        "**추가 값**은 모델에 없지만 자주 쓰이는 구간이다.",
        "- 폭은 fp32와 fp16 두 번 돈다. bf16은 백엔드가 거부한다(README 참조).",
        "",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(head + body))
    print(f"{len(ops())} ops -> {OUT}")


if __name__ == "__main__":
    main()
