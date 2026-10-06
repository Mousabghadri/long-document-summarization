import argparse
import json
import os
import time

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate a full long-document summary"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=2000,
    help="Target summary length in words"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=8000,
    help="Maximum source tokens per chunk"
)

parser.add_argument(
    "--output-name",
    required=True,
    help="Output name, e.g. ai_act"
)

args = parser.parse_args()

SUMMARY_WORDS = args.summary_words
CONTEXT_SIZE = args.context_size


# ============================================================
# HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


model_calls = 0


def call_model(prompt):
    global model_calls

    result = generate_summary(prompt)

    model_calls += 1

    if result is None:
        raise RuntimeError(
            "The model returned no text."
        )

    return result


# ============================================================
# LOAD DOCUMENT
# ============================================================

document_text = load_document(
    args.file
)

chunks = chunk_document(
    document_text,
    max_tokens=CONTEXT_SIZE
)

if not chunks:
    raise ValueError(
        "No chunks were created from the document."
    )

print("Total chunks:", len(chunks))
print("Context size:", CONTEXT_SIZE, "tokens")
print("Target summary:", SUMMARY_WORDS, "words")


# ============================================================
# LOAD PROMPTS
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

summarize_template = env.get_template(
    "summarize.j2"
)

update_template = env.get_template(
    "update.j2"
)

compress_template = env.get_template(
    "compress.j2"
)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

base_output_dir = os.path.join(
    "outputs",
    "full_summaries",
    args.output_name
)

intermediate_dir = os.path.join(
    base_output_dir,
    "intermediate"
)

os.makedirs(
    intermediate_dir,
    exist_ok=True
)


# ============================================================
# COMPRESSION
# ============================================================

def compress_to_target(
    summary,
    label
):

    max_attempts = 3
    attempt = 0

    while (
        count_words(summary) > SUMMARY_WORDS
        and attempt < max_attempts
    ):

        attempt += 1

        print(
            f"{label}: compression attempt {attempt}"
        )

        print(
            "Current words:",
            count_words(summary)
        )

        compress_prompt = compress_template.render(
            summary=summary,
            summary_length=SUMMARY_WORDS
        )

        summary = call_model(
            compress_prompt
        )

        print(
            "Words after compression:",
            count_words(summary)
        )

    return summary


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# FIRST CHUNK
# ============================================================

print(
    f"\nProcessing chunk 1/{len(chunks)}..."
)

first_prompt = summarize_template.render(
    chunk=chunks[0],
    summary_length=SUMMARY_WORDS
)

current_summary = call_model(
    first_prompt
)

current_summary = compress_to_target(
    current_summary,
    "Chunk 1"
)


# Save intermediate summary

with open(
    os.path.join(
        intermediate_dir,
        "chunk_1.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        current_summary
    )


# ============================================================
# REMAINING CHUNKS
# ============================================================

for i, chunk in enumerate(
    chunks[1:],
    start=2
):

    print(
        f"\nProcessing chunk {i}/{len(chunks)}..."
    )

    update_prompt = update_template.render(
        current_summary=current_summary,
        chunk=chunk,

        # Kept for compatibility with current template
        target_tokens=SUMMARY_WORDS,

        # Actual target for this experiment
        summary_length=SUMMARY_WORDS
    )

    current_summary = call_model(
        update_prompt
    )

    current_summary = compress_to_target(
        current_summary,
        f"Chunk {i}"
    )


    # Save intermediate summary

    intermediate_path = os.path.join(
        intermediate_dir,
        f"chunk_{i}.txt"
    )

    with open(
        intermediate_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            current_summary
        )

    print(
        "Current summary words:",
        count_words(current_summary)
    )


# ============================================================
# FINISH TIMER
# ============================================================

runtime = time.time() - start_time

final_word_count = count_words(
    current_summary
)


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_filename = (
    f"{args.output_name}_"
    f"{SUMMARY_WORDS}_words.txt"
)

summary_path = os.path.join(
    base_output_dir,
    summary_filename
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        current_summary
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "document": args.file,
    "target_words": SUMMARY_WORDS,
    "final_words": final_word_count,
    "context_size": CONTEXT_SIZE,
    "chunks": len(chunks),
    "model_calls": model_calls,
    "runtime_seconds": runtime
}

metadata_path = os.path.join(
    base_output_dir,
    "metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(
    "Final word count:",
    final_word_count
)

print(
    f"Runtime: {runtime:.2f} seconds"
)

print(
    "Model calls:",
    model_calls
)

print(
    "Final summary saved:",
    summary_path
)

print(
    "Metadata saved:",
    metadata_path
)