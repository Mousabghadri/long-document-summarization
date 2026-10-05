import argparse

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import (
    chunk_document,
    count_tokens,
    truncate_to_complete_sentence
)
from src.llm import generate_summary


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Sequential Long Document Summarization"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

parser.add_argument(
    "--summary-size",
    type=int,
    default=500,
    help="Target size of intermediate and final summaries in tokens"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=4000,
    help="Maximum number of source tokens sent to the model at a time"
)

args = parser.parse_args()

SUMMARY_SIZE = args.summary_size
CONTEXT_SIZE = args.context_size
SUMMARY_LENGTH_WORDS = int(SUMMARY_SIZE * 0.75)


# ============================================================
# LOAD DOCUMENT
# ============================================================

document_text = load_document(
    args.file
)


# ============================================================
# CREATE CHUNKS
# ============================================================

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
print("Target summary size:", SUMMARY_SIZE, "tokens")


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
# HELPER: COMPRESS SUMMARY
# ============================================================
def compress_to_target(
    summary,
    label="Summary"
):

    max_compression_attempts = 5
    attempt = 0

    while (
        count_tokens(summary) > SUMMARY_SIZE
        and attempt < max_compression_attempts
    ):

        attempt += 1

        current_tokens = count_tokens(summary)

        target_words = int(SUMMARY_SIZE * 0.75)

        compression_percentage = max(
            10,
            int(
                (SUMMARY_SIZE / current_tokens) * 100
            )
        )

        print(
            f"{label} exceeds target size -> "
            f"compression attempt {attempt}..."
        )

        print(
            f"Current tokens: {current_tokens} | "
            f"Target: {SUMMARY_SIZE} | "
            f"Target ratio: {compression_percentage}%"
        )

        compress_prompt = compress_template.render(
    summary=summary,
    summary_length=SUMMARY_LENGTH_WORDS
)

        summary = generate_summary(
            compress_prompt
        )

        if summary is None:
            raise RuntimeError(
                "The model returned no text during compression."
            )

        print(
            "Tokens after compression:",
            count_tokens(summary)
        )

    # Hard fallback if the model still exceeds the limit
    if count_tokens(summary) > SUMMARY_SIZE:

     print(
        f"Hard truncation applied at complete sentence: "
        f"{count_tokens(summary)} -> <= {SUMMARY_SIZE} tokens"
     )

     summary = truncate_to_complete_sentence(
        summary,
        SUMMARY_SIZE
     )

     print(
        "Tokens after sentence-safe truncation:",
        count_tokens(summary)
     )

    return summary
# ============================================================
# FIRST CHUNK
# ============================================================

first_prompt = summarize_template.render(
    chunk=chunks[0],
    summary_length=SUMMARY_LENGTH_WORDS
)

current_summary = generate_summary(
    first_prompt
)

if current_summary is None:
    raise RuntimeError(
        "The model returned no text for the first summary."
    )

current_summary = compress_to_target(
    current_summary,
    label="First summary"
)


print("\n" + "=" * 80)
print("SUMMARY AFTER CHUNK 1")
print("=" * 80)

print(current_summary)

print(
    "Summary tokens:",
    count_tokens(current_summary)
)


# ============================================================
# PROCESS REMAINING CHUNKS
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
        target_tokens=SUMMARY_SIZE
    )

    current_summary = generate_summary(
        update_prompt
    )

    if current_summary is None:
        raise RuntimeError(
            f"The model returned no text after chunk {i}."
        )

    summary_tokens = count_tokens(
        current_summary
    )

    print(
        "Current summary tokens:",
        summary_tokens
    )

    current_summary = compress_to_target(
        current_summary,
        label=f"Summary after chunk {i}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINAL SEQUENTIAL SUMMARY")
print("=" * 80)

print(current_summary)

print(
    "\nFinal summary tokens:",
    count_tokens(current_summary)
)