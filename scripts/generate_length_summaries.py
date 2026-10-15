
import argparse
import json
import os
import sys
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


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

    return result


model_calls = 0


def call_model(prompt, max_retries=3):
    global model_calls

    for attempt in range(1, max_retries + 1):

        try:
            result = generate_summary(prompt)
            model_calls += 1

            if result is not None and result.strip():
                return result.strip()

            print(
                f"Empty model response. Retry {attempt}/{max_retries}..."
            )

        except RuntimeError as e:

            print(
                f"Model error: {e}"
            )

            if attempt < max_retries:
                print(
                    f"Retrying... {attempt}/{max_retries}"
                )

                time.sleep(2)

    raise RuntimeError(
        "The model failed after multiple retries."
    )


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
    f"intermediate_{SUMMARY_WORDS}_words"
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

    max_attempts = 2

    for attempt in range(
        1,
        max_attempts + 1
    ):

        current_words = count_words(
            summary
        )

        if current_words <= SUMMARY_WORDS:
            break

        # If the summary exceeds the target only slightly,
        # avoid an extra LLM compression call.
        if current_words <= int(SUMMARY_WORDS * 1.10):

            print(
                f"{label}: small overshoot "
                f"({current_words} words), "
                "using deterministic sentence-safe truncation."
            )

            summary = truncate_to_word_limit(
                summary,
                SUMMARY_WORDS
            )

            print(
                "Words after deterministic truncation:",
                count_words(summary)
            )

            break

        print(
            f"{label}: compression attempt {attempt}"
        )

        print(
            "Current words:",
            current_words
        )

        compress_prompt = compress_template.render(
            summary=summary,
            summary_length=SUMMARY_WORDS
        )

        compressed = call_model(
            compress_prompt
        )

        compressed_words = count_words(
            compressed
        )

        print(
            "Words after compression:",
            compressed_words
        )

        # Accept compression only if it actually improves length
        if compressed_words < current_words:

            summary = compressed

        else:

            print(
                "Compression did not improve the length. Stopping compression."
            )

            break

    # Final deterministic fallback
    if count_words(summary) > SUMMARY_WORDS:

        print(
            f"{label}: applying final word-limit fallback..."
        )

        summary = truncate_to_word_limit(
            summary,
            SUMMARY_WORDS
        )

        print(
            "Words after fallback:",
            count_words(summary)
        )

    return summary
# ============================================================
# FIND SAVED PROGRESS
# ============================================================

def find_latest_intermediate():

    latest_chunk = 0
    latest_path = None

    for i in range(
        1,
        len(chunks) + 1
    ):

        path = os.path.join(
            intermediate_dir,
            f"chunk_{i}.txt"
        )

        if os.path.exists(path):

            latest_chunk = i
            latest_path = path

        else:
            break

    return latest_chunk, latest_path


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# RESUME OR START
# ============================================================

latest_chunk, latest_path = find_latest_intermediate()


if latest_chunk > 0:

    print(
        f"\nFound saved progress at chunk {latest_chunk}."
    )

    print(
        f"Resuming from chunk {latest_chunk + 1}..."
    )

    with open(
        latest_path,
        "r",
        encoding="utf-8"
    ) as f:

        current_summary = f.read()

    start_chunk_index = latest_chunk


else:

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

    print(
        "Current summary words:",
        count_words(current_summary)
    )

    start_chunk_index = 1


# ============================================================
# REMAINING CHUNKS
# ============================================================

for chunk_index in range(
    start_chunk_index,
    len(chunks)
):

    chunk_number = chunk_index + 1

    print(
        f"\nProcessing chunk {chunk_number}/{len(chunks)}..."
    )

    chunk = chunks[
        chunk_index
    ]

    update_prompt = update_template.render(
        current_summary=current_summary,
        chunk=chunk,
        target_tokens=SUMMARY_WORDS,
        summary_length=SUMMARY_WORDS
    )

    updated_summary = call_model(
        update_prompt
    )

    current_summary = compress_to_target(
        updated_summary,
        f"Chunk {chunk_number}"
    )


    # Save progress immediately after every successful chunk

    intermediate_path = os.path.join(
        intermediate_dir,
        f"chunk_{chunk_number}.txt"
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

    print(
        f"Progress saved: chunk {chunk_number}"
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
    "model_calls_this_run": model_calls,
    "runtime_seconds_this_run": runtime
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
    f"Runtime this run: {runtime:.2f} seconds"
)

print(
    "Model calls this run:",
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
