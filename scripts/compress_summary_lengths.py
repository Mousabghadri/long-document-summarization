import argparse
import os
import re
import sys
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import generate_summary


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Compress a long summary into smaller target lengths"
)

parser.add_argument(
    "file",
    help="Path to the source summary text file"
)

parser.add_argument(
    "--output-prefix",
    required=True,
    help="Output prefix, e.g. ai_act or gdpr"
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


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

            # Skip separator row
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

        # Remove bold / italic markers
        stripped = stripped.replace("**", "")
        stripped = stripped.replace("__", "")

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

    # Remove headings that appeared in the middle of a line
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


def truncate_to_complete_sentence_by_words(
    text,
    max_words
):
    if count_words(text) <= max_words:
        return text

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    kept_sentences = []

    for sentence in sentences:

        candidate = " ".join(
            kept_sentences + [sentence]
        )

        if count_words(candidate) <= max_words:
            kept_sentences.append(
                sentence
            )
        else:
            break

    if kept_sentences:
        result = " ".join(
            kept_sentences
        ).strip()

        # Remove incomplete numbered item like "32."
        result = re.sub(
            r"\n?\s*\d+\.\s*$",
            "",
            result
        ).strip()

        return result

    # Fallback only if first sentence itself is too long
    words = text.split()[:max_words]

    result = " ".join(words).strip()

    result = re.sub(
        r"\n?\s*\d+\.\s*$",
        "",
        result
    ).strip()

    return result


# ============================================================
# LOAD SOURCE SUMMARY
# ============================================================

with open(
    args.file,
    "r",
    encoding="utf-8"
) as f:
    current_summary = f.read().strip()


# Clean the input before doing anything else

current_summary = clean_plain_text(
    current_summary
)


print(
    "Starting summary words:",
    count_words(current_summary)
)


# ============================================================
# LOAD PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

compress_template = env.get_template(
    "compress.j2"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = os.path.dirname(
    args.file
)

if not output_dir:
    output_dir = "."


# ============================================================
# TARGET LENGTHS
# ============================================================

targets = [
    1000,
    500,
    250
]


# ============================================================
# PROCESS TARGETS
# ============================================================

total_model_calls = 0
total_start_time = time.time()


for target_words in targets:

    print("\n" + "=" * 80)
    print(
        f"COMPRESSING TO ~{target_words} WORDS"
    )
    print("=" * 80)

    start_time = time.time()

    compressed_summary = current_summary


    # --------------------------------------------------------
    # MODEL COMPRESSION - ATTEMPT 1
    # --------------------------------------------------------

    prompt = compress_template.render(
        summary=compressed_summary,
        summary_length=target_words
    )

    compressed_summary = generate_summary(
        prompt
    )

    total_model_calls += 1

    if compressed_summary is None or not compressed_summary.strip():
        raise RuntimeError(
            f"Model returned no text for {target_words}-word summary."
        )

    compressed_summary = clean_plain_text(
        compressed_summary
    )

    print(
        "Words after compression attempt 1:",
        count_words(compressed_summary)
    )


    # --------------------------------------------------------
    # MODEL COMPRESSION - ATTEMPT 2
    # --------------------------------------------------------

    if count_words(compressed_summary) > target_words:

        prompt = compress_template.render(
            summary=compressed_summary,
            summary_length=target_words
        )

        second_result = generate_summary(
            prompt
        )

        total_model_calls += 1

        if second_result is not None and second_result.strip():

            second_result = clean_plain_text(
                second_result
            )

            print(
                "Words after compression attempt 2:",
                count_words(second_result)
            )

            # Accept only if it actually improves length
            if count_words(second_result) < count_words(compressed_summary):
                compressed_summary = second_result
            else:
                print(
                    "Second compression did not improve length."
                )


    # --------------------------------------------------------
    # SENTENCE-SAFE FALLBACK
    # --------------------------------------------------------

    if count_words(compressed_summary) > target_words:

        before_truncation = count_words(
            compressed_summary
        )

        compressed_summary = (
            truncate_to_complete_sentence_by_words(
                compressed_summary,
                target_words
            )
        )

        print(
            "Sentence-safe fallback:",
            before_truncation,
            "->",
            count_words(compressed_summary),
            "words"
        )


    # Final formatting cleanup

    compressed_summary = clean_plain_text(
        compressed_summary
    )


    # ========================================================
    # FINAL STATS FOR THIS TARGET
    # ========================================================

    runtime = time.time() - start_time

    final_words = count_words(
        compressed_summary
    )

    print(
        "Final word count:",
        final_words
    )

    print(
        f"Runtime: {runtime:.2f} seconds"
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    output_path = os.path.join(
        output_dir,
        f"{args.output_prefix}_{target_words}_words.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            compressed_summary
        )

    print(
        "Saved:",
        output_path
    )


    # ========================================================
    # USE RESULT FOR NEXT LEVEL
    # ========================================================

    current_summary = compressed_summary


# ============================================================
# FINAL STATS
# ============================================================

total_runtime = (
    time.time() - total_start_time
)

print("\n" + "=" * 80)
print("COMPRESSION CHAIN COMPLETE")
print("=" * 80)

print(
    f"Total runtime: {total_runtime:.2f} seconds"
)

print(
    "Total model calls:",
    total_model_calls
)