
import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import count_tokens  # noqa: E402

OUT = PROJECT_ROOT / "data" / "datasets"


def safe_id(x):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))[:80]


def write_pair(ds, doc_id, text, summary):
    d = OUT / ds
    (d / "references").mkdir(parents=True, exist_ok=True)
    src = d / f"{doc_id}.txt"
    ref = d / "references" / f"{doc_id}.txt"
    src.write_text(text.strip(), encoding="utf-8")
    ref.write_text(summary.strip(), encoding="utf-8")
    return {
        "name": f"{ds}_{doc_id}",
        "dataset": ds,
        "source": str(src.relative_to(PROJECT_ROOT)),
        "reference": str(ref.relative_to(PROJECT_ROOT)),
        "source_tokens": count_tokens(text),
        "reference_words": len(summary.split()),
    }


def sample(items, n, seed, exclude, min_tokens, max_tokens):
    items = [it for it in items if it[0] not in exclude
             and it[1].strip() and it[2].strip()]
    rng = random.Random(seed)
    rng.shuffle(items)
    chosen = []
    for it in items:
        tokens = count_tokens(it[1])
        if tokens < min_tokens or (max_tokens and tokens > max_tokens):
            continue
        chosen.append(it)
        if len(chosen) >= n:
            break
    return chosen


def load_casesumm():
    from datasets import load_dataset
    ds = load_dataset("ChicagoHAI/CaseSumm", split="train")
    print("CaseSumm: only a train split is published; sampling from it with a fixed seed.")
    return [(safe_id(r["citation"] or i), r["opinion"] or "", r["syllabus"] or "")
            for i, r in enumerate(ds)]


def load_eurlexsum(splits="test"):
    from datasets import load_dataset
    rows = []
    for split in [x.strip() for x in splits.split(",") if x.strip()]:
        ds = load_dataset("dennlinger/eur-lex-sum", "english", split=split, trust_remote_code=True)
        rows += [(safe_id(r["celex_id"]), r["reference"] or "", r["summary"] or "") for r in ds]
    return rows


def load_inabs(root):
    root = Path(root).expanduser()
    judg = next((p for p in root.rglob("judgement") if p.is_dir()
                 and "IN-Abs" in str(p) and "test" in str(p)), None)
    if judg is None:
        sys.exit(f"Could not find IN-Abs/test-data/judgement under {root}")
    summ = judg.parent / "summary"
    items = []
    for f in sorted(judg.glob("*.txt")):
        s = summ / f.name
        if s.exists():
            items.append((safe_id(f.stem), f.read_text(encoding="utf-8", errors="replace"),
                          s.read_text(encoding="utf-8", errors="replace")))
    return items


def main():
    p = argparse.ArgumentParser(description="Prepare CaseSumm / IN-Abs / EUR-Lex-Sum samples")
    p.add_argument("--datasets", default="casesumm,eurlexsum,inabs")
    p.add_argument("--n", type=int, default=30, help="documents per dataset")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--inabs-dir", default=None)
    p.add_argument("--exclude-file", default=None, help="ids used for prompt development")
    p.add_argument("--min-tokens", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0, help="0 = no limit")
    p.add_argument("--eurlex-splits", default="test",
                   help="EUR-Lex-Sum splits to sample from, e.g. test,validation,train "
                        "(all are unseen for training-free methods)")
    p.add_argument("--out-name", default="", help="manifest file name suffix (default: dataset name)")
    a = p.parse_args()

    exclude = set()
    if a.exclude_file:
        exclude = {l.strip() for l in Path(a.exclude_file).read_text().splitlines() if l.strip()}

    loaders = {"casesumm": load_casesumm, "eurlexsum": lambda: load_eurlexsum(a.eurlex_splits),
               "inabs": lambda: load_inabs(a.inabs_dir or sys.exit("--inabs-dir is required"))}
    all_rows = []
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        if ds not in loaders:
            sys.exit(f"unknown dataset {ds}")
        chosen = sample(loaders[ds](), a.n, a.seed, exclude, a.min_tokens, a.max_tokens)
        rows = [write_pair(ds, i, t, s) for i, t, s in chosen]
        man = OUT / f"manifest_{a.out_name or ds}.jsonl"
        man.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        toks = sorted(r["source_tokens"] for r in rows)
        print(f"{ds}: {len(rows)} documents -> {man}  "
              f"(tokens min/median/max = {toks[0]}/{toks[len(toks) // 2]}/{toks[-1]})" if rows else
              f"{ds}: no documents selected")
        all_rows.extend(rows)

    if all_rows:
        man = OUT / "manifest_all.jsonl"
        man.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
        print("Combined manifest:", man)


if __name__ == "__main__":
    main()