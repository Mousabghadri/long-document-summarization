
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD_FILES = [
    "main.py",
    "scripts/aggregate_source_verification.py",
    "scripts/compress_summary_lengths.py",
    "scripts/create_method_lengths.py",
    "scripts/generate_claim_summary.py",
    "scripts/generate_support_summary.py",
    "scripts/repair_claim_summary.py",
    "scripts/show_problematic_verification.py",
    "scripts/summn_planning.py",
    "prompts/evaluate_quality.j2",
    "prompts/repair_claim_summary.j2",
    "prompts/verify_final_summary.j2",
    "data/dga.txt",
]

REPORT_TAGS = [
    "fair_v2", "day2_clean",
    "exp1b_eval", "exp1c_eval", "exp3_eval",
    "test_eurlex",
    "test_long_all",
    "models_eurlex",
    "mr_eval",
]


def tag_of(path_str):
    parts = re.split(r"[\\/]+", path_str)
    for i, part in enumerate(parts[:-1]):
        if part == "comparison" and i > 0 and parts[i - 1] == "outputs":
            return parts[i + 1]
    return None


def referenced_tags(comp, start):
    keep, todo = set(), list(start)
    while todo:
        tag = todo.pop()
        if tag in keep or not (comp / tag).is_dir():
            continue
        keep.add(tag)
        runs_file = comp / tag / "runs.json"
        if runs_file.exists():
            runs = json.loads(runs_file.read_text(encoding="utf-8"))
            for rec in runs.values():
                t = tag_of(rec.get("summary_path", "") or "")
                if t and t not in keep:
                    todo.append(t)
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()

    archive = ROOT / "archive"
    moves = [ROOT / f for f in OLD_FILES if (ROOT / f).exists()]

    comp = ROOT / "outputs" / "comparison"
    if comp.is_dir():
        keep = referenced_tags(comp, REPORT_TAGS)
        tags = sorted(d.name for d in comp.iterdir() if d.is_dir())
        print("Comparison tags kept (used by the report or referenced by it):")
        print("  " + ", ".join(t for t in tags if t in keep))
        moves += [comp / t for t in tags if t not in keep]

    if not moves:
        print("Nothing to move.")
        return
    print("\nTo archive/:")
    for m in moves:
        print("  " + str(m.relative_to(ROOT)))

    if not a.apply:
        print("\nDry run - nothing moved. Re-run with --apply to move these.")
        return
    for m in moves:
        dest = archive / m.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(m), str(dest))
    print(f"\nMoved {len(moves)} item(s) to {archive}")


if __name__ == "__main__":
    sys.exit(main())
