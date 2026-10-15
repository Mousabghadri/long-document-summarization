import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.llm import generate_summary


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Repair a claims-based summary against supported claims."
)

parser.add_argument(
    "claims_file",
    help="Path to supported claims file"
)

parser.add_argument(
    "summary_file",
    help="Path to draft summary file"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=1000,
    help="Maximum summary length"
)

parser.add_argument(
    "--output-file",
    required=True,
    help="Path to save repaired summary"
)

args = parser.parse_args()


# ============================================================
# LOAD FILES
# ============================================================

claims_text = Path(
    args.claims_file
).read_text(
    encoding="utf-8"
).strip()

summary_text = Path(
    args.summary_file
).read_text(
    encoding="utf-8"
).strip()


print(
    "Supported claims:",
    len(claims_text.splitlines())
)

print(
    "Draft summary words:",
    len(summary_text.split())
)


# ============================================================
# LOAD PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

template = env.get_template(
    "repair_claim_summary.j2"
)

prompt = template.render(
    claims_text=claims_text,
    summary_text=summary_text,
    summary_length=args.summary_words
)


# ============================================================
# REPAIR
# ============================================================

print()
print("Repairing summary...")

repaired_summary = generate_summary(
    prompt
)

if not repaired_summary:
    raise RuntimeError(
        "Model returned no repaired summary."
    )

repaired_summary = repaired_summary.strip()


# ============================================================
# SAVE
# ============================================================

output_path = Path(
    args.output_file
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    repaired_summary,
    encoding="utf-8"
)


# ============================================================
# RESULT
# ============================================================

print()
print(
    "Repaired summary words:",
    len(repaired_summary.split())
)

print(
    "Saved:",
    output_path
)