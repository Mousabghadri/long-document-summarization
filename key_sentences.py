import argparse
import os
import re
import time

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Extract and validate key sentences from a long document"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=4000,
    help="Maximum number of source tokens per chunk"
)

parser.add_argument(
    "--max-sentences",
    type=int,
    default=10,
    help="Maximum number of key sentences to extract per chunk"
)

parser.add_argument(
    "--max-chunks",
    type=int,
    default=3,
    help="Number of chunks to process for testing"
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

test_chunks = chunks[:args.max_chunks]

print("Total chunks:", len(chunks))
print("Processing chunks:", len(test_chunks))


# ============================================================
# LOAD PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

key_sentence_template = env.get_template(
    "extract_key_sentences.j2"
)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

base_output_dir = "outputs/key_sentences"

extracted_dir = os.path.join(
    base_output_dir,
    "extracted"
)

validated_dir = os.path.join(
    base_output_dir,
    "validated"
)

rejected_dir = os.path.join(
    base_output_dir,
    "rejected"
)

os.makedirs(
    extracted_dir,
    exist_ok=True
)

os.makedirs(
    validated_dir,
    exist_ok=True
)

os.makedirs(
    rejected_dir,
    exist_ok=True
)


# ============================================================
# START MEASUREMENT
# ============================================================

start_time = time.time()
model_calls = 0


# ============================================================
# PROCESS CHUNKS
# ============================================================

for i, chunk in enumerate(
    test_chunks,
    start=1
):

    print(
        f"\nProcessing chunk {i}..."
    )

    prompt = key_sentence_template.render(
        source_text=chunk,
        max_sentences=args.max_sentences
    )

    result = generate_summary(
        prompt
    )

    model_calls += 1


    # ========================================================
    # EXTRACT MODEL OUTPUT
    # ========================================================

    extracted_sentences = [
        line.strip()
        for line in result.splitlines()
        if line.strip()
    ]


    # ========================================================
    # NORMALIZE SOURCE
    # ========================================================

    normalized_chunk = normalize_text(
        chunk
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    validated_sentences = []
    rejected_sentences = []

    for sentence in extracted_sentences:

        normalized_sentence = normalize_text(
            sentence
        )

        if normalized_sentence in normalized_chunk:
            validated_sentences.append(
                sentence
            )
        else:
            rejected_sentences.append(
                sentence
            )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "Extracted:",
        len(extracted_sentences)
    )

    print(
        "Exact matches:",
        len(validated_sentences)
    )

    print(
        "Rejected:",
        len(rejected_sentences)
    )


    # ========================================================
    # SAVE EXTRACTED
    # ========================================================

    extracted_path = os.path.join(
        extracted_dir,
        f"chunk_{i}.txt"
    )

    with open(
        extracted_path,
        "w",
        encoding="utf-8"
    ) as f:

        for sentence in extracted_sentences:
            f.write(
                sentence + "\n"
            )


    # ========================================================
    # SAVE VALIDATED
    # ========================================================

    validated_path = os.path.join(
        validated_dir,
        f"chunk_{i}.txt"
    )

    with open(
        validated_path,
        "w",
        encoding="utf-8"
    ) as f:

        for sentence in validated_sentences:
            f.write(
                sentence + "\n"
            )


    # ========================================================
    # SAVE REJECTED
    # ========================================================

    rejected_path = os.path.join(
        rejected_dir,
        f"chunk_{i}.txt"
    )

    with open(
        rejected_path,
        "w",
        encoding="utf-8"
    ) as f:

        for sentence in rejected_sentences:
            f.write(
                sentence + "\n"
            )

    print(
        "Saved intermediate outputs for chunk",
        i
    )


# ============================================================
# END MEASUREMENT
# ============================================================

end_time = time.time()

runtime = end_time - start_time

print(
    f"\nTotal runtime: {runtime:.2f} seconds"
)

print(
    "Model calls:",
    model_calls
)