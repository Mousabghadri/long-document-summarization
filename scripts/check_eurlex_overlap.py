"""
Are the four development regulations part of EUR-Lex-Sum?  Also prints the
size of every English split (training-free methods can be evaluated on all).

    python -m scripts.check_eurlex_overlap
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import load_dataset  # noqa: E402

DEV = {"32016R0679": "GDPR", "32022R0868": "Data Governance Act",
       "32023R2854": "Data Act", "32024R1689": "AI Act"}

total = 0
found = {}
for split in ("train", "validation", "test"):
    ds = load_dataset("dennlinger/eur-lex-sum", "english", split=split, trust_remote_code=True)
    total += len(ds)
    print(f"{split:10s}: {len(ds):5d} documents")
    for r in ds:
        cid = r["celex_id"]
        if cid in DEV:
            found[cid] = (split, len((r["summary"] or "").split()))
print(f"total     : {total:5d} documents\n")
for cid, name in DEV.items():
    if cid in found:
        split, words = found[cid]
        print(f"{name:20s} {cid}: IN EUR-Lex-Sum ({split} split, reference {words} words)")
    else:
        print(f"{name:20s} {cid}: not in EUR-Lex-Sum")
