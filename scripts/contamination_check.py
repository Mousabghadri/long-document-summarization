"""
Does a method just pick up an existing summary?  Two checks, no LLM calls:

1. Where do the reference's words come from?  ROUGE recall of the reference
   against the act's preamble (citations + recitals) and against its operative
   part (articles).  If the recitals cover the reference much better per token,
   a method that reads the start of the document gets a head start.
2. Writes a copy of every document WITHOUT the preamble (title + articles only)
   plus a manifest, so Direct can be re-run on it:

    python -m scripts.contamination_check data/datasets/manifest_eurlex_test15.jsonl
    python compare_methods.py generate --tag norec --manifest data/datasets/manifest_eurlex_test15_norecitals.jsonl \
        --target-length reference --methods direct

The title-only memorization probe is the method `title_only` in compare_methods.py.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer  # noqa: E402

from src.chunking import count_tokens  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.eu_structure import split_act  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("manifest")
a = p.parse_args()

man = Path(a.manifest)
entries = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
out_dir = ROOT / "data" / "datasets" / f"{man.stem}_norecitals"
out_dir.mkdir(parents=True, exist_ok=True)


def resolve(x):
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


rows, new_entries = [], []
print("| Document | Preamble tokens | Articles tokens | R-1 recall: preamble | articles | "
      "R-2 recall: preamble | articles |")
print("|---|---:|---:|---:|---:|---:|---:|")
for e in entries:
    text = load_document(str(resolve(e["source"])))
    ref = Path(resolve(e["reference"])).read_text(encoding="utf-8")
    parts = split_act(text)
    if not parts["found"]:
        print(f"| {e['name']} | (no preamble/articles split found) | | | | | |")
        continue
    sp, so = scorer.score(ref, parts["preamble"]), scorer.score(ref, parts["operative"])
    r = (count_tokens(parts["preamble"]), count_tokens(parts["operative"]),
         sp["rouge1"].recall, so["rouge1"].recall, sp["rouge2"].recall, so["rouge2"].recall)
    rows.append(r)
    print(f"| {e['name']} | {r[0]:,} | {r[1]:,} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} |")
    dest = out_dir / f"{e['name']}.txt"
    dest.write_text(parts["title"] + "\n\n" + parts["operative"], encoding="utf-8")
    new_entries.append({**e, "source": str(dest.relative_to(ROOT))})

if rows:
    m = lambda i: statistics.fmean(r[i] for r in rows)
    print(f"| **mean** | {m(0):,.0f} | {m(1):,.0f} | {m(2):.3f} | {m(3):.3f} | {m(4):.3f} | {m(5):.3f} |")
    print(f"\nPreamble = {m(0) / (m(0) + m(1)):.0%} of the tokens on average.")
new_man = man.with_name(man.stem + "_norecitals.jsonl")
new_man.write_text("\n".join(json.dumps(x) for x in new_entries) + "\n", encoding="utf-8")
print(f"Documents without preamble: {out_dir}  ({len(new_entries)} of {len(entries)})")
print(f"Manifest: {new_man}")
