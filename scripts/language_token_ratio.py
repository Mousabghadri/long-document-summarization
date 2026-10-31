"""
Which EUR-Lex-Sum language tokenizes worst?  Token counts of the same documents
in several languages, relative to English (median ratio per document).

    $env:PYTHONUTF8=1; python -m scripts.language_token_ratio
    python -m scripts.language_token_ratio --langs german,greek,bulgarian --n 40

Uses the pipeline's tokenizer (tiktoken cl100k); the models' own tokenizers
differ, so read the ratios as relative, not exact.
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import load_dataset  # noqa: E402

from src.chunking import count_tokens  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--langs", default="german,french,polish,hungarian,finnish,greek,bulgarian")
p.add_argument("--n", type=int, default=30, help="documents per language (from the test split)")
a = p.parse_args()

def docs(lang):
    ds = load_dataset("dennlinger/eur-lex-sum", lang, split="test", trust_remote_code=True)
    return {r["celex_id"]: r["reference"] or "" for r in ds}

en = docs("english")
ids = [i for i in en if en[i].strip()][: a.n]
en_tok = {i: count_tokens(en[i]) for i in ids}
print(f"| Language | Docs | Median tokens / English | Median tokens per document |")
print("|---|---:|---:|---:|")
print(f"| english | {len(ids)} | 1.00 | {int(statistics.median(en_tok.values())):,} |")
for lang in [x.strip() for x in a.langs.split(",") if x.strip()]:
    try:
        d = docs(lang)
    except Exception as e:  # noqa: BLE001
        print(f"| {lang} | error: {str(e)[:60]} | | |")
        continue
    pairs = [(count_tokens(d[i]), en_tok[i]) for i in ids if i in d and d[i].strip()]
    if not pairs:
        print(f"| {lang} | 0 | - | - |")
        continue
    ratios = [t / max(e, 1) for t, e in pairs]
    print(f"| {lang} | {len(pairs)} | {statistics.median(ratios):.2f} | "
          f"{int(statistics.median(t for t, _ in pairs)):,} |")
