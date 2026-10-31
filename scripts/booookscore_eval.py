import argparse
import json
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm import generate_summary  # noqa: E402
from src.postprocess import split_sentences  # noqa: E402


LABELS = [
    "entity omission",
    "event omission",
    "causal omission",
    "salience",
    "discontinuity",
    "duplication",
    "inconsistency",
    "language"
]


p = argparse.ArgumentParser()

p.add_argument("summary_file")

p.add_argument(
    "--output-file",
    required=True
)

p.add_argument(
    "--template",
    default=str(ROOT / "prompts" / "booookscore_official.txt")
)

p.add_argument(
    "--workers",
    type=int,
    default=5,
    help="parallel annotator calls (per-sentence mode)"
)

p.add_argument(
    "--model",
    default="deepseek/deepseek-v3.2"
)

p.add_argument(
    "--batch-size",
    type=int,
    default=10
)

a = p.parse_args()


summary = Path(a.summary_file).read_text(
    encoding="utf-8"
).strip()

template = Path(a.template).read_text(
    encoding="utf-8"
)

sentences = [
    s
    for s in (
        x.strip()
        for line in summary.splitlines()
        for x in split_sentences(line)
    )
    if len(s.split()) >= 3
]

if not sentences:
    sys.exit("summary has no sentences")


def norm_types(t):

    t = (t or "").lower()

    if "no confusion" in t:
        return []

    return [
        x.strip()
        for x in t.replace(";", ",").split(",")
        if x.strip() in LABELS
    ]


annots = []

if "{sentences}" in template:

    for start in range(0, len(sentences), a.batch_size):

        batch = sentences[start:start + a.batch_size]

        prompt = template.format(
            summary=summary,
            sentences="\n".join(
                f"{i + 1}. {s}"
                for i, s in enumerate(batch)
            )
        )

        parsed = None

        for _ in range(3):

            out = generate_summary(
                prompt,
                model_name=a.model,
                purpose="booookscore"
            )

            text = out.strip()

            i, j = text.find("["), text.rfind("]")

            try:
                parsed = (
                    json.loads(text[i:j + 1])
                    if i != -1 and j > i
                    else None
                )

            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, list) and len(parsed) == len(batch):
                break

            parsed = None

        if parsed is None:
            sys.exit(
                f"annotator did not return {len(batch)} answers "
                f"for sentences {start + 1}-"
            )

        for s, ans in zip(batch, parsed):

            ans = ans if isinstance(ans, dict) else {}

            annots.append({
                "sentence": s,
                "questions": ans.get("questions"),
                "types": norm_types(ans.get("types"))
            })

else:

    from concurrent.futures import ThreadPoolExecutor

    def annotate(s):

        for _ in range(3):

            out = generate_summary(
                template.format(summary, s),
                model_name=a.model,
                purpose="booookscore"
            )

            lines = [
                ln
                for ln in out.splitlines()
                if "Questions:" in ln or "Types:" in ln
            ]

            if len(lines) >= 2:

                q = lines[0].split("Questions:", 1)[1].strip()

                return {
                    "sentence": s,
                    "questions": q,
                    "types": norm_types(
                        lines[1].split("Types:", 1)[1]
                    )
                }

        raise RuntimeError(
            "annotator answer not in 'Questions: / Types:' format: "
            + s[:60]
        )

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:

        annots = list(
            pool.map(annotate, sentences)
        )


clear = sum(
    1 for x in annots
    if not x["types"]
)

counts = {
    lab: sum(lab in x["types"] for x in annots)
    for lab in LABELS
}

result = {
    "booookscore": round(clear / len(annots), 4),
    "sentences": len(annots),
    "confusing_sentences": len(annots) - clear,
    "error_counts": counts,
    "template": Path(a.template).name,
    "model": a.model,
    "annotations": annots
}


Path(a.output_file).parent.mkdir(
    parents=True,
    exist_ok=True
)

Path(a.output_file).write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print(
    f"BooookScore: {result['booookscore']:.3f} "
    f"({clear}/{len(annots)} sentences without confusion)"
)