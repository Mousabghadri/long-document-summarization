import argparse
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.llm import generate_summary


# ============================================================
# HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


def truncate_to_word_limit(text, max_words):
    sentences = text.split(". ")

    selected = []
    total_words = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        words = sentence.split()

        if total_words + len(words) > max_words:
            break

        selected.append(sentence)
        total_words += len(words)

    result = ". ".join(selected).strip()

    if result and not result.endswith("."):
        result += "."

    # Remove incomplete numbered item at the end, e.g. "32."
    result = re.sub(
        r"\n?\s*\d+\.\s*$",
        "",
        result
    ).strip()

    return result


def clean_plain_text(text):
    """
    Convert common Markdown formatting to plain text
    without changing factual content.
    """

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Remove horizontal rules
        if re.fullmatch(r"-{3,}", stripped):
            continue

        # Remove Markdown heading markers
        stripped = re.sub(
            r"^#{1,6}\s*",
            "",
            stripped
        )

        # Convert Markdown tables to plain text
        if stripped.startswith("|") and stripped.endswith("|"):

            cells = [
                cell.strip()
                for cell in stripped.strip("|").split("|")
            ]

            # Skip Markdown table separator rows
            if all(
                re.fullmatch(r":?-+:?", cell)
                for cell in cells
            ):
                continue

            if len(cells) == 2:
                stripped = f"{cells[0]}: {cells[1]}"
            else:
                stripped = " - ".join(cells)

        # Convert Markdown bullets to plain text bullets
        stripped = re.sub(
            r"^[-*+]\s+",
            "- ",
            stripped
        )

        # Remove bold markers
        stripped = stripped.replace("**", "")
        stripped = stripped.replace("__", "")

        # Remove remaining italic markers
        stripped = re.sub(
            r"(?<!\*)\*(?!\*)",
            "",
            stripped
        )

        stripped = re.sub(
            r"(?<!_)_(?!_)",
            "",
            stripped
        )

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Remove Markdown headings that appear in the middle of a line
    text = re.sub(
        r"\s+#{1,6}\s+",
        "\n\n",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate and compress a final summary from supported claims"
)

parser.add_argument(
    "claims_file",
    help="Path to supported claims txt file"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=2000
)

parser.add_argument(
    "--additional-instructions",
    type=str,
    default=""
)

parser.add_argument(
    "--output-file",
    required=True
)

args = parser.parse_args()


# ============================================================
# LOAD CLAIMS
# ============================================================

claims_text = Path(
    args.claims_file
).read_text(
    encoding="utf-8"
)


# ============================================================
# LOAD PROMPTS
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

summary_template = env.get_template(
    "summarize_from_claims.j2"
)

compress_template = env.get_template(
    "compress.j2"
)

verify_template = env.get_template(
    "verify_final_summary.j2"
)


# ============================================================
# STEP 1: GENERATE SUMMARY
# ============================================================

prompt = summary_template.render(
    claims_text=claims_text,
    summary_length=args.summary_words,
    additional_instructions=args.additional_instructions
)

print("Generating summary...")

summary = generate_summary(
    prompt
)

if summary is None or not summary.strip():
    raise RuntimeError(
        "No summary generated."
    )


# Force plain text immediately

summary = clean_plain_text(
    summary
)

print(
    "Initial summary words:",
    count_words(summary)
)


# ============================================================
# STEP 2: COMPRESS IF TOO LONG
# ============================================================

max_attempts = 2
attempt = 0

while (
    count_words(summary) > args.summary_words
    and attempt < max_attempts
):

    attempt += 1

    print(
        f"Compressing summary... attempt {attempt}"
    )

    compress_prompt = compress_template.render(
        summary=summary,
        summary_length=args.summary_words
    )

    compressed = generate_summary(
        compress_prompt
    )

    if compressed is None or not compressed.strip():
        print(
            "Compression returned no text."
        )
        break

    compressed = clean_plain_text(
        compressed
    )

    summary = compressed

    print(
        "Compressed words:",
        count_words(summary)
    )


# ============================================================
# STEP 3: FINAL WORD-LIMIT FALLBACK
# ============================================================

if count_words(summary) > args.summary_words:

    print(
        "Applying final word-limit fallback..."
    )

    summary = truncate_to_word_limit(
        summary,
        args.summary_words
    )


# Final formatting safety check

summary = clean_plain_text(
    summary
)


# ============================================================
# STEP 4: VERIFY FINAL SUMMARY
# ============================================================

print("Verifying final summary...")

verify_prompt = verify_template.render(
    claims_text=claims_text,
    summary_text=summary
)

verification = generate_summary(
    verify_prompt
)

if verification is None:

    print(
        "Final verification failed."
    )

else:

    verification_path = Path(
        args.output_file
    ).with_name(
        Path(args.output_file).stem
        + "_verification.json"
    )

    verification_path.write_text(
        verification,
        encoding="utf-8"
    )

    print(
        "Verification saved:",
        verification_path
    )


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

output_path = Path(
    args.output_file
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    summary,
    encoding="utf-8"
)

print(
    "Final summary words:",
    count_words(summary)
)

print(
    "Saved:",
    output_path
)