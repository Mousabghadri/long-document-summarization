import argparse
import os
import re
import time

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


parser = argparse.ArgumentParser(
    description="Extract validated key sentences from a full document"
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
    help="Maximum key sentences per chunk"
)

parser.add_argument(
    "--output-name",
    required=True,
    help="Output name, e.g. ai_act or gdpr"
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


# ============================================================
# LOAD PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

template = env.get_template(
    "extract_key_sentences.j2"
)


# ============================================================
# OUTPUT
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

all_validated_sentences = []

model_calls = 0
start_time = time.time()


# ============================================================
# PROCESS ALL CHUNKS
# ============================================================

for i, chunk in enumerate(
    chunks,
    start=1
):

    print(
        f"\nProcessing chunk {i}/{len(chunks)}..."
    )

    prompt = template.render(
        source_text=chunk,
        max_sentences=args.max_sentences
    )

    result = generate_summary(
        prompt
    )

    model_calls += 1

    if result is None:
        raise RuntimeError(
            f"No model output for chunk {i}"
        )

    extracted_sentences = [
        line.strip()
        for line in result.splitlines()
        if line.strip()
    ]

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

    print(
        "Extracted:",
        len(extracted_sentences)
    )

    print(
        "Validated:",
        len(validated_sentences)
    )

    all_validated_sentences.extend(
        validated_sentences
    )


# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

output_path = os.path.join(
    output_dir,
    f"{args.output_name}_key_sentences.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    for sentence in all_validated_sentences:
        f.write(
            sentence + "\n"
        )


runtime = time.time() - start_time

print("\n" + "=" * 80)
print("KEY SENTENCE EXTRACTION COMPLETE")
print("=" * 80)

print(
    "Validated sentences:",
    len(all_validated_sentences)
)

print(
    f"Runtime: {runtime:.2f} seconds"
)

print(
    "Model calls:",
    model_calls
)

print(
    "Saved:",
    output_path
)