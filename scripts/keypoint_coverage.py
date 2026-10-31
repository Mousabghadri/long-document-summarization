import argparse
import hashlib
import json
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm import generate_summary  # noqa: E402
from src.postprocess import prompt_env  # noqa: E402


CACHE = ROOT / "outputs" / "keypoints"


def parse_list(text):

    i, j = text.find("["), text.rfind("]")

    if i == -1 or j <= i:
        return None

    try:
        return json.loads(text[i:j + 1])

    except json.JSONDecodeError:
        return None


def key_points(reference, model):

    CACHE.mkdir(
        parents=True,
        exist_ok=True
    )

    path = CACHE / (
        hashlib.sha1(reference.encode("utf-8")).hexdigest()[:16]
        + ".json"
    )

    if path.exists():

        return json.loads(
            path.read_text(encoding="utf-8")
        )["points"]

    prompt = prompt_env().get_template(
        "keypoints_extract.j2"
    ).render(
        reference=reference
    )

    for _ in range(3):

        pts = parse_list(
            generate_summary(
                prompt,
                model_name=model,
                purpose="keypoints_extract"
            )
        )

        if isinstance(pts, list) and pts and all(
            isinstance(p, str) for p in pts
        ):
            break

        pts = None

    if not pts:
        sys.exit("key-point extraction failed")

    tmp = path.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(
            {"model": model, "points": pts},
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    tmp.replace(path)

    return pts


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "reference"
    )

    p.add_argument(
        "summary",
        nargs="?"
    )

    p.add_argument(
        "--output-file"
    )

    p.add_argument(
        "--model",
        default="deepseek/deepseek-v3.2"
    )

    p.add_argument(
        "--extract-only",
        action="store_true"
    )

    a = p.parse_args()


    reference = Path(a.reference).read_text(
        encoding="utf-8"
    ).strip()

    points = key_points(
        reference,
        a.model
    )

    if a.extract_only:

        print(f"{len(points)} key points")

        return


    summary = Path(a.summary).read_text(
        encoding="utf-8"
    ).strip()

    prompt = prompt_env().get_template(
        "keypoints_check.j2"
    ).render(
        summary=summary,
        points="\n".join(
            f"{i + 1}. {pt}"
            for i, pt in enumerate(points)
        )
    )

    labels = None

    for _ in range(3):

        out = parse_list(
            generate_summary(
                prompt,
                model_name=a.model,
                purpose="keypoints_check"
            )
        )

        if isinstance(out, list) and len(out) == len(points):

            labels = [
                str((x or {}).get("label", "missing")).lower()
                if isinstance(x, dict)
                else "missing"
                for x in out
            ]

            break

    if labels is None:
        sys.exit("key-point check failed")

    labels = [
        lab if lab in ("covered", "partial", "missing") else "missing"
        for lab in labels
    ]

    covered, partial = (
        labels.count("covered"),
        labels.count("partial")
    )

    score = (covered + 0.5 * partial) / len(points)

    res = {
        "keypoint_coverage": round(score, 4),
        "points": len(points),
        "covered": covered,
        "partial": partial,
        "missing": labels.count("missing"),
        "model": a.model,
        "details": [
            {"point": pt, "label": lab}
            for pt, lab in zip(points, labels)
        ]
    }

    Path(a.output_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(a.output_file).write_text(
        json.dumps(
            res,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"Key-point coverage: {score:.3f} "
        f"({covered} covered, {partial} partial of {len(points)})"
    )


if __name__ == "__main__":
    main()