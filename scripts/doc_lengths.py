import argparse
import json
import statistics
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import count_tokens  # noqa: E402
from src.document_loader import load_document  # noqa: E402


p = argparse.ArgumentParser()

p.add_argument(
    "manifests",
    nargs="+"
)

p.add_argument(
    "--per-doc",
    action="store_true"
)

p.add_argument(
    "--context",
    type=int,
    default=100000,
    help="context limit used by Direct"
)

a = p.parse_args()


def resolve(path):

    path = Path(path)

    return path if path.is_absolute() else ROOT / path


rows, docs = [], []


for man in a.manifests:

    entries = [
        json.loads(l)
        for l in Path(man).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    toks, refw = [], []

    for e in entries:

        t = count_tokens(
            load_document(
                str(resolve(e["source"]))
            )
        )

        r = len(
            Path(resolve(e["reference"]))
            .read_text(encoding="utf-8")
            .split()
        )

        toks.append(t)
        refw.append(r)

        docs.append((
            Path(man).stem,
            e["name"],
            t,
            r
        ))

    over = sum(
        t > a.context
        for t in toks
    )

    rows.append((
        Path(man).stem,
        len(toks),
        min(toks),
        int(statistics.median(toks)),
        int(statistics.fmean(toks)),
        max(toks),
        over,
        min(refw),
        int(statistics.median(refw)),
        max(refw),
        int(
            statistics.median([
                t / max(r, 1)
                for t, r in zip(toks, refw)
            ])
        )
    ))


print(
    f"| Set | Docs | Tokens min | median | mean | max | "
    f"> {a.context // 1000}K | "
    "Reference words min | median | max | "
    "Compression (tokens per reference word, median) |"
)

print(
    "|---|---:|---:|---:|---:|---:|---:|"
    "---:|---:|---:|---:|"
)


for r in rows:

    print(
        "| "
        + " | ".join(
            f"{x:,}" if isinstance(x, int) else str(x)
            for x in r
        )
        + " |"
    )


if a.per_doc:

    print(
        "\n| Set | Document | Tokens | Reference words |\n"
        "|---|---|---:|---:|"
    )

    for s, n, t, r in sorted(
        docs,
        key=lambda x: (x[0], x[2])
    ):

        print(
            f"| {s} | {n} | {t:,} | {r:,} |"
        )