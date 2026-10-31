"""
Merge the runs of several tags into one tag, so they are evaluated in the SAME
run (LLM-judge scores drift between runs; only compare within one evaluation).

    python -m scripts.merge_tags --out models_eurlex test_eurlex oss_mr_eurlex gemma_eurlex:@gemma
    python -m scripts.merge_tags --out exp1c_eval exp1b_eval=key_sentences_overview exp1c_fast

Each source is  TAG[=method1,method2][:@suffix]
  =methods   keep only these methods from that tag
  :@suffix   rename its methods to method@suffix (e.g. runs of another model);
             the report labels them "Method [suffix]"
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "outputs" / "comparison"

p = argparse.ArgumentParser()
p.add_argument("--out", required=True, help="new tag")
p.add_argument("sources", nargs="+", help="TAG[=methods][:@suffix]")
a = p.parse_args()

merged = {}
for src in a.sources:
    suffix = ""
    if ":@" in src:
        src, suffix = src.split(":@", 1)
        suffix = "@" + suffix
    keep = None
    if "=" in src:
        src, methods = src.split("=", 1)
        keep = {m.strip() for m in methods.split(",") if m.strip()}
    path = COMP / src / "runs.json"
    if not path.exists():
        sys.exit(f"not found: {path}")
    runs = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for rec in runs.values():
        if keep and rec["method"] not in keep:
            continue
        rec = dict(rec, method=rec["method"] + suffix)
        merged[f"{rec['doc']}|{rec['method']}|{rec['rep']}"] = rec
        n += 1
    print(f"{src}{suffix or ''}: {n} runs")

out = COMP / a.out
out.mkdir(parents=True, exist_ok=True)
(out / "runs.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
ok = sum(r.get("status") == "ok" for r in merged.values())
print(f"-> {out / 'runs.json'}: {len(merged)} runs, {ok} ok")
