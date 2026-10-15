import argparse
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def count_words(text):
    return len(text.split())


parser = argparse.ArgumentParser(
    description="Extract validated support sentences and generate a grounded summary"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=8000,
    help="Maximum source tokens per chunk"
)

parser.add_argument(
    "--max-sentences",
    type=int,
    default=10,
    help="Maximum support sentences per chunk"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=2000,
    help="Target final summary length in words"
)

parser.add_argument(
    "--output-name",
    required=True,
    help="Output name, e.g. ai_act or gdpr"
)

parser.add_argument(
    "--additional-instructions",
    type=str,
    default="",
    help="Optional extra instructions for summary focus"
)

args = parser.parse_args()


# ============================================================
# LOAD DOCUMENT
# ============================================================

document_text = load_document(
    args.file
)

chunks = chunk_document(
    document_text,
    max_tokens=args.context_size
)

print("Total chunks:", len(chunks))
print("Context size:", args.context_size)
print("Target summary:", args.summary_words, "words")


# ============================================================
# LOAD PROMPTS
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

extract_template = env.get_template(
    "extract_key_sentences.j2"
)

summary_template = env.get_template(
    "summarize_from_support.j2"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = os.path.join(
    "outputs",
    "method_outputs",
    args.output_name
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# STATS
# ============================================================

all_validated_sentences = []

total_extracted = 0
total_validated = 0

model_calls = 0
start_time = time.time()


# ============================================================
# STEP 1: EXTRACT + VALIDATE SUPPORT SENTENCES
# ============================================================

for i, chunk in enumerate(
    chunks,
    start=1
):

    print(
        f"\nProcessing chunk {i}/{len(chunks)}..."
    )

    prompt = extract_template.render(
        source_text=chunk,
        max_sentences=args.max_sentences
    )

    result = None
    max_retries = 3

    for attempt in range(
        1,
        max_retries + 1
    ):
        try:
            result = generate_summary(
                prompt
            )

            model_calls += 1

            if result is not None and result.strip():
                break

            print(
                f"Chunk {i}: empty response "
                f"(attempt {attempt}/{max_retries})"
            )

        except RuntimeError as e:
            model_calls += 1

            print(
                f"Chunk {i}: model error "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

        if attempt < max_retries:
            time.sleep(2)

    if result is None or not result.strip():
        raise RuntimeError(
            f"No model output for chunk {i} "
            f"after {max_retries} attempts"
        )

    extracted_sentences = [
        line.strip()
        for line in result.splitlines()
        if line.strip()
    ]

    total_extracted += len(
        extracted_sentences
    )

    normalized_chunk = normalize_text(
        chunk
    )

    validated_sentences = []

    for sentence in extracted_sentences:

        normalized_sentence = normalize_text(
            sentence
        )

        if normalized_sentence in normalized_chunk:
            validated_sentences.append(
                sentence
            )

    total_validated += len(
        validated_sentences
    )

    all_validated_sentences.extend(
        validated_sentences
    )

    print(
        "Extracted:",
        len(extracted_sentences)
    )

    print(
        "Validated:",
        len(validated_sentences)
    )


# ============================================================
# SAVE SUPPORT SENTENCES
# ============================================================

support_path = os.path.join(
    output_dir,
    f"{args.output_name}_support_sentences.txt"
)

with open(
    support_path,
    "w",
    encoding="utf-8"
) as f:

    for sentence in all_validated_sentences:
        f.write(
            sentence + "\n"
        )


# ============================================================
# STEP 2: GENERATE FINAL SUMMARY
# ============================================================

support_text = "\n".join(
    all_validated_sentences
)

print(
    "\nGenerating final summary from validated support..."
)

summary_prompt = summary_template.render(
    support_text=support_text,
    summary_length=args.summary_words,
    additional_instructions=args.additional_instructions
)

final_summary = generate_summary(
    summary_prompt
)

model_calls += 1

if final_summary is None or not final_summary.strip():
    raise RuntimeError(
        "No final summary was generated."
    )


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_path = os.path.join(
    output_dir,
    f"{args.output_name}_key_sentences_summary_{args.summary_words}_words.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        final_summary
    )


# ============================================================
# FINAL STATS
# ============================================================

runtime = time.time() - start_time

validation_rate = (
    total_validated / total_extracted * 100
    if total_extracted > 0
    else 0
)

print("\n" + "=" * 80)
print("EXTRACT-SUPPORT SUMMARIZATION COMPLETE")
print("=" * 80)

print(
    "Extracted sentences:",
    total_extracted
)

print(
    "Validated support sentences:",
    total_validated
)

print(
    f"Validation rate: {validation_rate:.2f}%"
)

print(
    "Final summary words:",
    count_words(final_summary)
)

print(
    f"Runtime: {runtime:.2f} seconds"
)

print(
    "Model calls:",
    model_calls
)

print(
    "Support saved:",
    support_path
)

print(
    "Summary saved:",
    summary_path
)