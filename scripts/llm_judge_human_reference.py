import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.llm import generate_summary


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="LLM-as-a-Judge evaluation against a human reference."
)

parser.add_argument(
    "human_reference_file",
    help="Path to human reference summary"
)

parser.add_argument(
    "generated_summary_file",
    help="Path to generated summary"
)

parser.add_argument(
    "--output-file",
    required=True,
    help="Path to save judge result JSON"
)

parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Optional model name to use for the LLM judge"
)

args = parser.parse_args()

# ============================================================
# LOAD FILES
# ============================================================

human_reference = Path(
    args.human_reference_file
).read_text(
    encoding="utf-8"
).strip()

generated_summary = Path(
    args.generated_summary_file
).read_text(
    encoding="utf-8"
).strip()


print(
    "Human reference words:",
    len(human_reference.split())
)

print(
    "Generated summary words:",
    len(generated_summary.split())
)


# ============================================================
# LOAD PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

template = env.get_template(
    "llm_judge_human_reference.j2"
)

prompt = template.render(
    human_reference=human_reference,
    generated_summary=generated_summary
)


# ============================================================
# CALL MODEL
# ============================================================

print()
print("Running LLM-as-a-Judge...")

result = generate_summary(
    prompt,
    model_name=args.model
)

if not result:
    raise RuntimeError(
        "Model returned no judge result."
    )

result = result.strip()
if result.startswith("```"):
    result = result.strip("`").strip()

    if result.lower().startswith("json"):
        result = result[4:].strip()


# ============================================================
# PARSE JSON
# ============================================================

try:
    parsed = json.loads(
        result
    )

except json.JSONDecodeError:

    print()
    print("Model returned invalid JSON.")
    print()
    print(result)

    raise


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
    json.dumps(
        parsed,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 70)
print("LLM JUDGE RESULT")
print("=" * 70)

print(
    "Coverage:",
    parsed.get("coverage_score")
)

print(
    "Emphasis:",
    parsed.get("emphasis_score")
)

print(
    "Reference alignment:",
    parsed.get("reference_alignment_score")
)

print(
    "Relevance:",
    parsed.get("relevance_score")
)

print(
    "Coherence:",
    parsed.get("coherence_score")
)

print(
    "Overall:",
    parsed.get("overall_score")
)

print()
print("Verdict:")
print(
    parsed.get("verdict")
)

print()
print(
    "Saved:",
    output_path
)