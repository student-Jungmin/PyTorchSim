#!/usr/bin/env python3
"""Turn TOGSim's default info-level logs into a self-contained NPU utilization page.

Reads `togsim_results/*.log` -- no re-run and no `--log_level trace` needed, since
the periodic `Core stat` blocks every `core_stats_print_period_cycles` already carry
the whole time series. For per-instruction Gantt detail use `trace_timeline.py`.

Usage:
  python scripts/util_viewer.py togsim_results -o util.html
  python scripts/util_viewer.py togsim_results/2026*.log -o util.html
"""
import argparse
import glob
import json
import os
import re
import sys

_CMDLINE = re.compile(r"--trace_so (\S+)")
_CONFIG_INT = re.compile(r"^(\w+): (\d+)$")
_PEAK_BW = re.compile(r"Total bandwidth ([\d.]+) GB/s, (\d+) MHz, (\d+) channels, (\d+) bytes")
_SA = re.compile(r"Core \[(\d+)\] : Systolic array \[(\d+)\] utilization\(%\): ([\d.]+), "
                 r"active_cycles: (\d+), idle_cycles: (\d+)")
_DMA = re.compile(r"Core \[(\d+)\] : DMA active_cycles: (\d+), DMA idle_cycles: (\d+), "
                  r"DRAM BW: ([\d.]+) GB/s \((\d+) responses\)")
_VU = re.compile(r"Core \[(\d+)\] : Vector unit [Uu]tilization\(%\): ([\d.]+), "
                 r"active[ _]cycles?: (\d+), idle[ _]cycles?: (\d+)")
_TOTCYC = re.compile(r"Core \[(\d+)\] : Total_cycles: (\d+)")
_INST = re.compile(r"Core \[(\d+)\] : (\w+) +inst_count: (\d+)(?: \(GEMM: (\d+), Vector: (\d+)\))?")
_DRAM_CH = re.compile(r"\[DRAM\] channel (\d+) \| ([\d.]+) GB/s avg\., ([\d.]+)% of utilization "
                      r"\| (\d+) reads, (\d+) writes")
_DRAM_WINDOW = re.compile(r"\[DRAM\] all \d+ channels combined \| ([\d.]+) GB/s "
                          r"aggregate, ([\d.]+)% of utilization")
_DRAM_TOTAL = re.compile(r"\[DRAM\] channels \d+\.\.\d+ combined \| ([\d.]+) GB/s "
                         r"aggregate, ([\d.]+)% of utilization")
_KERNEL_DONE = re.compile(r"\[Scheduler (\d+)\] Kernel (\d+) .* finished at cycle (\d+)")
_TOTAL_EXEC = re.compile(r"Total execution cycles: (\d+)")

_FINAL_MARKERS = ("=== DRAM statistics ===", "[DRAM] Per-channel average bandwidth",
                  "===== Instructions count =====")

_WANTED_CONFIG = ("num_cores", "core_freq_mhz", "core_stats_print_period_cycles",
                  "num_systolic_array_per_core", "vpu_num_lanes", "vpu_vector_length_bits",
                  "vpu_spad_size_kb_per_lane", "dram_channels", "dram_freq_mhz",
                  "core_spad_size_kb", "pytorchsim_timing_mode", "pytorchsim_functional_mode")


def kernel_name(trace_so):
    """Name a kernel by its `triton_<hash>` dir and the test dir above it."""
    parts = [p for p in trace_so.split(os.sep) if p]
    kdir = next((p for p in reversed(parts) if p.startswith("triton_")), None)
    if kdir is None:
        kdir = parts[-2] if len(parts) > 1 else trace_so
    i = parts.index(kdir) if kdir in parts else -1
    source = parts[i - 1] if i > 0 else ""
    if source in ("outputs", ".triton_sweep", "scratchpad"):
        source = ""
    return kdir, source


def _blank_core():
    return {"sa": {}, "vu": None, "dma": None, "cycles": 0, "inst": {}}


def _core(cores, cid):
    return cores.setdefault(int(cid), _blank_core())


def parse_log(path):
    """Parse one TOGSim info-level log into a run dict, or None if it has no stats."""
    with open(path, errors="replace") as fh:
        text = fh.read()
    if "Total execution cycles" not in text:
        return None

    cut = min((text.index(m) for m in _FINAL_MARKERS if m in text), default=len(text))
    head, tail = text[:cut], text[cut:]
    run = {"log": os.path.basename(path), "config": {}, "windows": [],
           "cores": {}, "dram_channels": [], "kernels": []}

    m = _CMDLINE.search(head)
    run["trace_so"] = m.group(1) if m else ""
    run["kernel"], run["source"] = kernel_name(run["trace_so"]) if m else ("?", "")

    for line in head.splitlines():
        cm = _CONFIG_INT.match(line.strip())
        if cm and cm.group(1) in _WANTED_CONFIG:
            run["config"].setdefault(cm.group(1), int(cm.group(2)))
    m = _PEAK_BW.search(head)
    if m:
        run["config"]["peak_dram_gbps"] = float(m.group(1))
        run["config"]["dram_req_bytes"] = int(m.group(4))

    interval = run["config"].get("core_stats_print_period_cycles", 0)
    cur = None
    for line in head.splitlines():
        if "========= Core stat =========" in line:
            cur = {"sa": {}, "vu": {}, "dma": {}, "dram_gbps": None, "dram_util": None}
            run["windows"].append(cur)
            continue
        if cur is None:
            continue
        m = _SA.search(line)
        if m:
            cur["sa"].setdefault(int(m.group(1)), {})[int(m.group(2))] = float(m.group(3))
            continue
        m = _VU.search(line)
        if m:
            cur["vu"][int(m.group(1))] = float(m.group(2))
            continue
        m = _DMA.search(line)
        if m:
            cur["dma"][int(m.group(1))] = {
                "util": 100.0 * int(m.group(2)) / interval if interval else 0.0,
                "gbps": float(m.group(4)), "responses": int(m.group(5))}
            continue
        m = _DRAM_WINDOW.search(line)
        if m:
            cur["dram_gbps"], cur["dram_util"] = float(m.group(1)), float(m.group(2))
    for m in _KERNEL_DONE.finditer(head):
        run["kernels"].append({"id": int(m.group(2)), "end": int(m.group(3))})

    for m in _INST.finditer(tail):
        c = _core(run["cores"], m.group(1))["inst"]
        c[m.group(2)] = int(m.group(3))
        if m.group(4) is not None:
            c["GEMM"], c["VECTOR"] = int(m.group(4)), int(m.group(5))
    for m in _SA.finditer(tail):
        _core(run["cores"], m.group(1))["sa"][int(m.group(2))] = {
            "util": float(m.group(3)), "active": int(m.group(4))}
    for m in _VU.finditer(tail):
        _core(run["cores"], m.group(1))["vu"] = {"util": float(m.group(2)), "active": int(m.group(3))}
    for m in _DMA.finditer(tail):
        _core(run["cores"], m.group(1))["dma"] = {
            "active": int(m.group(2)), "gbps": float(m.group(4)), "responses": int(m.group(5))}
    for m in _TOTCYC.finditer(tail):
        _core(run["cores"], m.group(1))["cycles"] = int(m.group(2))
    for m in _DRAM_CH.finditer(tail):
        run["dram_channels"].append({"ch": int(m.group(1)), "gbps": float(m.group(2)),
                                     "util": float(m.group(3)), "reads": int(m.group(4)),
                                     "writes": int(m.group(5))})
    m = _DRAM_TOTAL.search(tail)
    if m:
        run["dram_gbps"], run["dram_util"] = float(m.group(1)), float(m.group(2))
    m = _TOTAL_EXEC.search(tail)
    run["cycles"] = int(m.group(1)) if m else 0

    for cid, c in run["cores"].items():
        c["dma"] = c["dma"] or {"active": 0, "gbps": 0.0, "responses": 0}
        c["dma"]["util"] = 100.0 * c["dma"]["active"] / c["cycles"] if c["cycles"] else 0.0
    run["interval"] = interval
    return run


def summarize(run):
    """Fold a run's per-core totals into the numbers the page leads with."""
    cores = run["cores"]
    n = max(1, len(cores))
    sa = [s["util"] for c in cores.values() for s in c["sa"].values()]
    vu = [c["vu"]["util"] for c in cores.values() if c["vu"]]
    dma = [c["dma"]["util"] for c in cores.values()]
    inst = {}
    for c in cores.values():
        for k, v in c["inst"].items():
            inst[k] = inst.get(k, 0) + v
    freq = run["config"].get("core_freq_mhz", 0)
    return {"cycles": run["cycles"],
            "us": run["cycles"] / (freq * 1.0) if freq else 0.0,
            "sa": sum(sa) / len(sa) if sa else 0.0, "sa_peak": max(sa) if sa else 0.0,
            "vu": sum(vu) / len(vu) if vu else 0.0,
            "dma": sum(dma) / len(dma) if dma else 0.0,
            "dram_gbps": run.get("dram_gbps", 0.0), "dram_util": run.get("dram_util", 0.0),
            "inst": inst, "cores": n}


def series(run):
    """Return the per-window utilization series, averaged across cores."""
    out = {"cycle": [], "sa": [], "vu": [], "dma": [], "dram": [], "gbps": []}
    for i, w in enumerate(run["windows"]):
        sa = [v for per in w["sa"].values() for v in per.values()]
        vu = list(w["vu"].values())
        dma = [d["util"] for d in w["dma"].values()]
        out["cycle"].append((i + 1) * run["interval"])
        out["sa"].append(round(sum(sa) / len(sa), 2) if sa else 0.0)
        out["vu"].append(round(sum(vu) / len(vu), 2) if vu else 0.0)
        out["dma"].append(round(sum(dma) / len(dma), 2) if dma else 0.0)
        out["dram"].append(w["dram_util"] or 0.0)
        out["gbps"].append(w["dram_gbps"] or 0.0)
    return out


def collect(paths):
    """Expand dirs and globs into parsed runs, newest log last."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.log")))
        else:
            files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    runs = []
    for f in sorted(set(files)):
        try:
            r = parse_log(f)
        except Exception as exc:
            print(f"skip {f}: {exc}", file=sys.stderr)
            continue
        if r:
            runs.append(r)
    return runs


def build_payload(runs):
    """Shape parsed runs into the JSON the page renders from, collapsing replays."""
    items, seen = [], {}
    for r in runs:
        key = (r["kernel"], r["cycles"])
        if key in seen:
            prev = seen[key]
            prev["repeats"] += 1
            prev["source"] = prev["source"] or r["source"]
            continue
        item = {"log": r["log"], "kernel": r["kernel"], "source": r["source"], "repeats": 1,
                "config": r["config"], "summary": summarize(r), "series": series(r),
                "dram_channels": r["dram_channels"], "interval": r["interval"],
                "cores": {str(k): {"sa": v["sa"], "vu": v["vu"], "dma": v["dma"],
                                   "cycles": v["cycles"], "inst": v["inst"]}
                          for k, v in r["cores"].items()}}
        seen[key] = item
        items.append(item)
    items.sort(key=lambda i: -i["summary"]["cycles"])
    return {"runs": items, "logs": len(runs)}


_PAGE_HEAD = """<title>TOGSim Utilization</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --track: #ececE6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --track: #2c2c2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --plane: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --track: #2c2c2a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 32px 24px 64px;
}
.wrap { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--ink-2); margin: 0 0 28px; font-size: 13px; }
.card {
  background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px;
  padding: 18px 20px; margin-bottom: 20px;
}
.card > h2 {
  font-size: 13px; font-weight: 600; margin: 0 0 2px; letter-spacing: 0.01em;
}
.card > .note { color: var(--muted); font-size: 12px; margin: 0 0 16px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px;
        background: var(--ring); border-radius: 8px; overflow: hidden; }
.kpi { background: var(--surface-1); padding: 14px 16px; }
.kpi .lab { color: var(--muted); font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.04em; margin-bottom: 6px; }
.kpi .val { font-size: 26px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; }
.kpi .val small { font-size: 13px; font-weight: 500; color: var(--ink-2); margin-left: 2px; }
.kpi .sub2 { color: var(--ink-2); font-size: 12px; margin-top: 4px;
             font-variant-numeric: tabular-nums; }
.meter { height: 6px; border-radius: 3px; background: var(--track); margin-top: 8px;
         overflow: hidden; }
.meter > i { display: block; height: 100%; border-radius: 3px; background: var(--s1); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 0 0 10px; font-size: 12px;
          color: var(--ink-2); }
.legend b { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
            margin-right: 6px; vertical-align: -1px; }
.chart-scroll { overflow-x: auto; }
svg { display: block; }
svg text { fill: var(--muted); font-size: 11px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.04em; cursor: pointer; user-select: none; }
th:first-child, td:first-child { text-align: left; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: color-mix(in srgb, var(--s1) 7%, transparent); }
tbody tr.on { background: color-mix(in srgb, var(--s1) 12%, transparent); }
td .k { font-family: ui-monospace, monospace; font-size: 12px; }
td .src { color: var(--muted); font-size: 11px; }
.bar { display: inline-block; width: 46px; height: 6px; border-radius: 3px;
       background: var(--track); vertical-align: 1px; margin-right: 7px; overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--s1); border-radius: 3px; }
.tbl-scroll { overflow-x: auto; max-height: 460px; overflow-y: auto; }
#tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
       background: var(--surface-1); border: 1px solid var(--ring); border-radius: 8px;
       padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.14);
       z-index: 9; font-variant-numeric: tabular-nums; }
#tip .th { font-weight: 600; margin-bottom: 5px; font-size: 11px; color: var(--ink-2); }
#tip .r { display: flex; gap: 10px; justify-content: space-between; }
#tip .r b { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
            margin-right: 5px; vertical-align: 0; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px 2px 7px;
        border-radius: 999px; border: 1px solid var(--ring); font-size: 11px;
        color: var(--ink-2); white-space: nowrap; }
.chip i { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.empty { color: var(--muted); font-size: 13px; padding: 8px 0; }
.cfg { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 12px; color: var(--ink-2); }
.cfg span b { color: var(--muted); font-weight: 500; }
</style>"""


def render(payload):
    """Emit the self-contained HTML page for a parsed payload."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _PAGE_HEAD + "\n" + _PAGE_BODY.replace("__DATA__", data)


_PAGE_BODY = r"""
<div class="wrap">
  <h1>TOGSim Utilization</h1>
  <p class="sub" id="subtitle"></p>

  <div class="card">
    <h2>Selected kernel</h2>
    <p class="note" id="sel-note"></p>
    <div class="kpis" id="kpis"></div>
    <div class="cfg" id="cfg" style="margin-top:16px"></div>
  </div>

  <div class="card">
    <h2>Utilization over time</h2>
    <p class="note" id="tl-note"></p>
    <div class="legend" id="legend"></div>
    <div class="chart-scroll"><svg id="tl" width="1040" height="270"></svg></div>
  </div>

  <div class="card">
    <h2>DRAM bandwidth over time</h2>
    <p class="note" id="bw-note"></p>
    <div class="chart-scroll"><svg id="bw" width="1040" height="180"></svg></div>
  </div>

  <div class="card">
    <h2>Kernels</h2>
    <p class="note">Click a row to load it above. Sorted by cycles; click a header to re-sort.</p>
    <div class="tbl-scroll">
      <table id="tbl">
        <thead><tr>
          <th data-k="kernel">Kernel</th><th data-k="bound">Bound by</th>
          <th data-k="cycles">Cycles</th><th data-k="us">Time (us)</th>
          <th data-k="sa">Systolic</th><th data-k="vu">Vector</th><th data-k="dma">DMA</th>
          <th data-k="dram_util">DRAM</th><th data-k="gbps">GB/s</th><th data-k="comp">COMP</th>
          <th data-k="movin">MOVIN</th><th data-k="movout">MOVOUT</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>DRAM channels</h2>
    <p class="note" id="ch-note"></p>
    <div class="chart-scroll"><svg id="ch" width="1040" height="170"></svg></div>
  </div>
</div>
<div id="tip"></div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const RUNS = DATA.runs;
const NS = 'http://www.w3.org/2000/svg';
const tip = document.getElementById('tip');
const fmt = n => n.toLocaleString('en-US');
const f1 = n => n.toFixed(1);
const SERIES = [
  {k:'sa',   label:'Systolic array', v:'--s1'},
  {k:'vu',   label:'Vector unit',    v:'--s2'},
  {k:'dma',  label:'DMA engine',     v:'--s3'},
  {k:'dram', label:'DRAM bandwidth', v:'--s4'},
];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let sel = 0, sortKey = 'cycles', sortDir = -1;

function el(p, tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  p.appendChild(n); return n;
}
function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }

function rowOf(r) {
  const s = r.summary, i = s.inst || {};
  return {kernel: r.kernel, source: r.source, cycles: s.cycles, us: s.us, sa: s.sa,
          vu: s.vu, dma: s.dma, dram_util: s.dram_util, gbps: s.dram_gbps,
          comp: i.COMP || 0, movin: i.MOVIN || 0, movout: i.MOVOUT || 0,
          gemm: i.GEMM || 0, vector: i.VECTOR || 0, bound: classify(s).name, s: s};
}

function classify(s) {
  const cands = [{n: 'systolic-bound', v: s.sa, c: '--s1'},
                 {n: 'vector-bound', v: s.vu, c: '--s2'},
                 {n: 'DRAM-bound', v: s.dram_util, c: '--s4'}];
  const top = cands.reduce((a, b) => (b.v > a.v ? b : a));
  if (top.v < 20) return {name: 'stalled', color: '--muted', top: top.n, v: top.v};
  return {name: top.n, color: top.c, top: top.n, v: top.v};
}

function chip(s) {
  const c = classify(s);
  return `<span class="chip"><i style="background:var(${c.color})"></i>${c.name}</span>`;
}

function kpi(lab, val, unit, sub, pct, hue) {
  const d = document.createElement('div'); d.className = 'kpi';
  d.innerHTML = `<div class="lab">${lab}</div><div class="val">${val}` +
                (unit ? `<small>${unit}</small>` : '') + `</div>` +
                (sub ? `<div class="sub2">${sub}</div>` : '') +
                (pct == null ? '' :
                  `<div class="meter"><i style="width:${Math.min(100,pct).toFixed(1)}%;` +
                  `background:var(${hue})"></i></div>`);
  return d;
}

function drawKPIs() {
  const r = RUNS[sel], s = r.summary, i = s.inst || {};
  const box = document.getElementById('kpis'); clear(box);
  const peak = r.config.peak_dram_gbps || 0;
  box.appendChild(kpi('Total cycles', fmt(s.cycles), '',
    s.us ? `${f1(s.us)} us @ ${r.config.core_freq_mhz} MHz` : '', null));
  box.appendChild(kpi('Systolic array', f1(s.sa), '%',
    `peak array ${f1(s.sa_peak)}%`, s.sa, '--s1'));
  box.appendChild(kpi('Vector unit', f1(s.vu), '%',
    `${fmt(i.VECTOR || 0)} vector ops`, s.vu, '--s2'));
  box.appendChild(kpi('DMA engine', f1(s.dma), '%',
    `${fmt(i.MOVIN || 0)} in / ${fmt(i.MOVOUT || 0)} out`, s.dma, '--s3'));
  box.appendChild(kpi('DRAM', f1(s.dram_gbps), 'GB/s',
    peak ? `${f1(s.dram_util)}% of ${f1(peak)} GB/s peak` : `${f1(s.dram_util)}% of peak`,
    s.dram_util, '--s4'));

  const c = r.config, cfg = document.getElementById('cfg');
  const rows = [['cores', c.num_cores], ['systolic arrays/core', c.num_systolic_array_per_core],
    ['VPU lanes', c.vpu_num_lanes], ['vector bits', c.vpu_vector_length_bits],
    ['spad KB/lane', c.vpu_spad_size_kb_per_lane], ['DRAM channels', c.dram_channels],
    ['timing mode', c.pytorchsim_timing_mode], ['functional mode', c.pytorchsim_functional_mode]];
  cfg.innerHTML = rows.filter(x => x[1] != null)
    .map(x => `<span><b>${x[0]}</b> ${x[1]}</span>`).join('');
  document.getElementById('sel-note').innerHTML =
    chip(s) + ' &nbsp;' + r.kernel + (r.source ? ' &nbsp;·&nbsp; ' + r.source : '') +
    ' &nbsp;·&nbsp; ' + r.log + (r.repeats > 1 ? ` &nbsp;·&nbsp; ${r.repeats} identical runs` : '');
}

function axes(svg, W, H, P, xmax, ymax, yunit, xlabel, xticks) {
  const x = v => P.l + (W - P.l - P.r) * (xmax ? v / xmax : 0);
  const y = v => H - P.b - (H - P.t - P.b) * (ymax ? v / ymax : 0);
  for (let i = 0; i <= 4; i++) {
    const v = ymax * i / 4;
    el(svg, 'line', {x1: P.l, x2: W - P.r, y1: y(v), y2: y(v),
                     stroke: i ? css('--grid') : css('--axis'), 'stroke-width': 1});
    el(svg, 'text', {x: P.l - 8, y: y(v) + 4, 'text-anchor': 'end'},
       (ymax >= 20 ? Math.round(v) : v.toFixed(1)) + (i === 4 ? yunit : ''));
  }
  if (xticks !== false) {
    for (let i = 0; i <= 5; i++) {
      const v = xmax * i / 5;
      el(svg, 'text', {x: x(v), y: H - P.b + 16, 'text-anchor': i === 0 ? 'start' :
                       (i === 5 ? 'end' : 'middle')},
         v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' :
                    (v >= 1e3 ? Math.round(v / 1e3) + 'k' : Math.round(v)));
    }
  }
  el(svg, 'text', {x: (P.l + W - P.r) / 2, y: H - 2, 'text-anchor': 'middle'}, xlabel);
  return {x, y};
}

function hover(svg, W, H, P, sc, cyc, rows, unit) {
  const line = el(svg, 'line', {y1: P.t, y2: H - P.b, stroke: css('--axis'),
                                'stroke-width': 1, opacity: 0});
  const dots = rows.map(r => el(svg, 'circle', {r: 4, fill: r.color, stroke: css('--surface-1'),
                                                'stroke-width': 2, opacity: 0}));
  svg.addEventListener('mousemove', ev => {
    const b = svg.getBoundingClientRect();
    const px = (ev.clientX - b.left) * (W / b.width);
    if (!cyc.length) return;
    let bi = 0, bd = Infinity;
    cyc.forEach((c, i) => { const d = Math.abs(sc.x(c) - px); if (d < bd) { bd = d; bi = i; } });
    const gx = sc.x(cyc[bi]);
    line.setAttribute('x1', gx); line.setAttribute('x2', gx); line.setAttribute('opacity', 1);
    rows.forEach((r, j) => {
      dots[j].setAttribute('cx', gx); dots[j].setAttribute('cy', sc.y(r.data[bi]));
      dots[j].setAttribute('opacity', 1);
    });
    tip.innerHTML = `<div class="th">cycle ${fmt(cyc[bi])}</div>` + rows.map(r =>
      `<div class="r"><span><b style="background:${r.color}"></b>${r.label}</span>` +
      `<span>${f1(r.data[bi])}${unit}</span></div>`).join('');
    tip.style.opacity = 1;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 10) + 'px';
    tip.style.top = Math.max(8, ev.clientY - tip.offsetHeight - 12) + 'px';
  });
  svg.addEventListener('mouseleave', () => {
    tip.style.opacity = 0; line.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
  });
}

function drawTimeline() {
  const r = RUNS[sel], s = r.series, svg = document.getElementById('tl');
  clear(svg);
  const W = 1040, H = 270, P = {l: 44, r: 92, t: 12, b: 30};
  const note = document.getElementById('tl-note');
  const leg = document.getElementById('legend');
  leg.innerHTML = SERIES.map(x =>
    `<span><b style="background:var(${x.v})"></b>${x.label}</span>`).join('');
  if (!s.cycle.length) {
    note.textContent = `No samples: this kernel ran ${fmt(r.summary.cycles)} cycles, ` +
      `inside a single ${fmt(r.interval)}-cycle sampling window. Lower ` +
      `core_stats_print_period_cycles in the config and re-run to sample it.`;
    return;
  }
  note.textContent = `${s.cycle.length} samples, one per ${fmt(r.interval)} cycles, ` +
    `averaged across ${r.summary.cores} core(s).`;
  const xmax = s.cycle[s.cycle.length - 1];
  const sc = axes(svg, W, H, P, xmax, 100, '%', 'cycle');
  const rows = SERIES.map(x => ({label: x.label, color: css(x.v), data: s[x.k]}));
  rows.forEach(row => {
    const d = row.data.map((v, i) => `${i ? 'L' : 'M'}${sc.x(s.cycle[i]).toFixed(1)} ` +
                                     `${sc.y(v).toFixed(1)}`).join(' ');
    el(svg, 'path', {d, fill: 'none', stroke: row.color, 'stroke-width': 2,
                     'stroke-linejoin': 'round', 'stroke-linecap': 'round'});
    const last = row.data[row.data.length - 1];
    el(svg, 'text', {x: W - P.r + 8, y: sc.y(last) + 4, fill: row.color,
                     'font-weight': 600}, f1(last) + '%');
  });
  hover(svg, W, H, P, sc, s.cycle, rows, '%');
}

function drawBW() {
  const r = RUNS[sel], s = r.series, svg = document.getElementById('bw');
  clear(svg);
  const W = 1040, H = 180, P = {l: 44, r: 92, t: 12, b: 30};
  const note = document.getElementById('bw-note');
  const peak = r.config.peak_dram_gbps || 0;
  if (!s.cycle.length) {
    note.textContent = `No samples: this kernel ran inside a single ` +
      `${fmt(r.interval)}-cycle sampling window. Run totals are above.`;
    return;
  }
  note.textContent = peak ? `Aggregate across ${r.config.dram_channels} channels; ` +
    `peak is ${f1(peak)} GB/s.` : 'Aggregate across all channels.';
  const xmax = s.cycle[s.cycle.length - 1];
  const ymax = Math.max(peak || 0, ...s.gbps) || 1;
  const sc = axes(svg, W, H, P, xmax, ymax, ' GB/s', 'cycle');
  if (peak) {
    el(svg, 'line', {x1: P.l, x2: W - P.r, y1: sc.y(peak), y2: sc.y(peak),
                     stroke: css('--muted'), 'stroke-width': 1, 'stroke-dasharray': '4 4'});
    el(svg, 'text', {x: W - P.r + 8, y: sc.y(peak) + 4}, 'peak');
  }
  const row = {label: 'DRAM bandwidth', color: css('--s4'), data: s.gbps};
  const d = s.gbps.map((v, i) => `${i ? 'L' : 'M'}${sc.x(s.cycle[i]).toFixed(1)} ` +
                                 `${sc.y(v).toFixed(1)}`).join(' ');
  el(svg, 'path', {d, fill: 'none', stroke: row.color, 'stroke-width': 2,
                   'stroke-linejoin': 'round', 'stroke-linecap': 'round'});
  hover(svg, W, H, P, sc, s.cycle, [row], ' GB/s');
}

function drawChannels() {
  const r = RUNS[sel], svg = document.getElementById('ch');
  clear(svg);
  const ch = r.dram_channels || [];
  const note = document.getElementById('ch-note');
  if (!ch.length) { note.textContent = 'No per-channel statistics in this log.'; return; }
  const spread = Math.max(...ch.map(c => c.util)) - Math.min(...ch.map(c => c.util));
  note.textContent = `${ch.length} channels, run-average utilization. ` +
    `Spread across channels ${f1(spread)} points.`;
  const W = 1040, H = 170, P = {l: 44, r: 16, t: 12, b: 30};
  const ymax = Math.max(10, ...ch.map(c => c.util));
  const sc = axes(svg, W, H, P, ch.length, ymax, '%', 'DRAM channel', false);
  const bw = (W - P.l - P.r) / ch.length;
  const every = ch.length > 24 ? 4 : (ch.length > 12 ? 2 : 1);
  ch.forEach((c, i) => {
    const h = Math.max(1, (H - P.b) - sc.y(c.util));
    const g = el(svg, 'rect', {x: P.l + i * bw + 1, y: sc.y(c.util), width: Math.max(1, bw - 2),
                               height: h, rx: 4, fill: css('--s1')});
    if (i % every === 0) {
      el(svg, 'text', {x: P.l + i * bw + bw / 2, y: H - P.b + 16, 'text-anchor': 'middle'}, c.ch);
    }
    g.addEventListener('mousemove', ev => {
      tip.innerHTML = `<div class="th">channel ${c.ch}</div>` +
        `<div class="r"><span>utilization</span><span>${f1(c.util)}%</span></div>` +
        `<div class="r"><span>bandwidth</span><span>${f1(c.gbps)} GB/s</span></div>` +
        `<div class="r"><span>reads</span><span>${fmt(c.reads)}</span></div>` +
        `<div class="r"><span>writes</span><span>${fmt(c.writes)}</span></div>`;
      tip.style.opacity = 1;
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 10) + 'px';
      tip.style.top = Math.max(8, ev.clientY - tip.offsetHeight - 12) + 'px';
    });
    g.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
  });
}

function bar(v, hue) {
  return `<span class="bar"><i style="width:${Math.min(100, v).toFixed(1)}%;` +
         `background:var(${hue})"></i></span>${f1(v)}%`;
}

function drawTable() {
  const tb = document.querySelector('#tbl tbody');
  const idx = RUNS.map((r, i) => ({i, r: rowOf(r)}));
  idx.sort((a, b) => {
    const x = a.r[sortKey], y = b.r[sortKey];
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * sortDir;
  });
  tb.innerHTML = idx.map(({i, r}) =>
    `<tr data-i="${i}" class="${i === sel ? 'on' : ''}">` +
    `<td><span class="k">${r.kernel}</span>` +
      (r.source ? `<br><span class="src">${r.source}</span>` : '') + `</td>` +
    `<td style="text-align:left">${chip(r.s)}</td>` +
    `<td>${fmt(r.cycles)}</td><td>${f1(r.us)}</td>` +
    `<td>${bar(r.sa, '--s1')}</td><td>${bar(r.vu, '--s2')}</td>` +
    `<td>${bar(r.dma, '--s3')}</td><td>${bar(r.dram_util, '--s4')}</td>` +
    `<td>${f1(r.gbps)}</td><td>${fmt(r.comp)}</td>` +
    `<td>${fmt(r.movin)}</td><td>${fmt(r.movout)}</td></tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => {
    sel = +tr.dataset.i; drawAll();
  }));
}

function drawAll() { drawKPIs(); drawTimeline(); drawBW(); drawChannels(); drawTable(); }

document.querySelectorAll('#tbl th').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k;
  sortDir = (k === sortKey) ? -sortDir : (k === 'kernel' ? 1 : -1);
  sortKey = k; drawTable();
}));

const totalCycles = RUNS.reduce((a, r) => a + r.summary.cycles, 0);
const replays = DATA.logs - RUNS.length;
document.getElementById('subtitle').textContent =
  `${RUNS.length} distinct kernels from ${DATA.logs} TOGSim logs` +
  (replays > 0 ? ` (${replays} identical replays collapsed)` : '') +
  `, ${fmt(totalCycles)} cycles total. ` +
  `Utilization is active cycles over total cycles, as TOGSim counts them.`;
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
drawAll();
</script>
"""


def main(argv=None):
    """Parse the given logs and write the viewer page."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=["togsim_results"],
                    help="log files, globs, or directories (default: togsim_results)")
    ap.add_argument("-o", "--out", default="util.html", help="output HTML path")
    args = ap.parse_args(argv)

    runs = collect(args.paths or ["togsim_results"])
    if not runs:
        print("no TOGSim logs with statistics found", file=sys.stderr)
        return 1
    with open(args.out, "w") as fh:
        fh.write(render(build_payload(runs)))
    print(f"{len(runs)} run(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
