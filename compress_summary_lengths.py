import argparse
import os
import re
import time

from jinja2 import Environment, FileSystemLoader

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
# LOAD SOURCE SUMMARY
# ============================================================

with open(
    args.file,
    "r",
    encoding="utf-8"
) as f:
    current_summary = f.read().strip()


# ============================================================
# HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


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
        return " ".join(
            kept_sentences
        )

    # Fallback only if the first sentence itself is too long
    words = text.split()[:max_words]

    return " ".join(words)


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

    if compressed_summary is None:
        raise RuntimeError(
            f"Model returned no text for {target_words}-word summary."
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

        compressed_summary = generate_summary(
            prompt
        )

        total_model_calls += 1

        if compressed_summary is None:
            raise RuntimeError(
                f"Model returned no text for {target_words}-word summary."
            )

        print(
            "Words after compression attempt 2:",
            count_words(compressed_summary)
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