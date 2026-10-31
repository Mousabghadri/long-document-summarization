
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


tag = sys.argv[1]
base = ROOT / "outputs" / "comparison" / tag
runs = load(base / "runs.json") or {}
rows = defaultdict(list)
for r in runs.values():
    if r.get("status") != "ok":
        continue
    ed = base / "eval" / re.sub(r"[^A-Za-z0-9_]", "_", r["doc"]) / f"{r['method']}_r{r['rep']}"
    std, cov = load(ed / "standard.json") or {}, load(ed / "coverage.json") or {}
    judges = [load(p) for p in sorted(ed.glob("judge_*.json")) if not p.name.endswith("_raw_debug.json")]
    judges = [j for j in judges if j and j.get("overall_score") is not None]
    rows[r["method"]].append({
        "ratio": (r.get("words") or 0) / max(r.get("target_words") or 1, 1),
        "trunc": r.get("fits_context") is False,
        "rouge": std.get("rouge_lsum_f1"), "kp": cov.get("keypoint_coverage"),
        "jcov": statistics.fmean(j["coverage_score"] for j in judges) if judges else None,
        "jall": statistics.fmean(j["overall_score"] for j in judges) if judges else None,
    })

print(f"## {tag}\n")
print("| Method | Summaries | Words / target (mean) | below 90% of target | truncated input | "
      "rho(KP coverage, judge coverage) | rho(ROUGE-Lsum, judge coverage) | rho(KP coverage, length) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
allrows = []
for m in sorted(rows):
    xs = rows[m]
    allrows += xs
    pair = lambda a, b: [(x[a], x[b]) for x in xs if x[a] is not None and x[b] is not None]
    kj, rj, kl = pair("kp", "jcov"), pair("rouge", "jcov"), pair("kp", "ratio")
    f = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"| {m} | {len(xs)} | {statistics.fmean(x['ratio'] for x in xs):.2f} | "
          f"{sum(x['ratio'] < 0.9 for x in xs)} | {sum(x['trunc'] for x in xs)} | "
          f"{f(spearman(*zip(*kj)) if kj else None)} | {f(spearman(*zip(*rj)) if rj else None)} | "
          f"{f(spearman(*zip(*kl)) if kl else None)} |")

print("\nKey-point coverage by summary length (words / target):\n")
bands = [(0, 0.8), (0.8, 0.9), (0.9, 1.01), (1.01, 99)]
print("| Method | " + " | ".join(f"{lo:.2f}-{hi:.2f}" if hi < 99 else f">{lo:.2f}" for lo, hi in bands) + " |")
print("|---|" + "---:|" * len(bands))
for m in sorted(rows):
    cells = []
    for lo, hi in bands:
        v = [x["kp"] for x in rows[m] if x["kp"] is not None and lo <= x["ratio"] < hi]
        cells.append(f"{statistics.fmean(v):.3f} (n={len(v)})" if v else "-")
    print(f"| {m} | " + " | ".join(cells) + " |")
