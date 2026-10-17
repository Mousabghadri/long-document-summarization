import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def count_words(text):
    return len(text.split())


def clean_plain_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if re.fullmatch(r"-{3,}", stripped):
            continue

        stripped = re.sub(
            r"^#{1,6}\s*",
            "",
            stripped
        )

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [
                cell.strip()
                for cell in stripped.strip("|").split("|")
            ]

            if all(
                re.fullmatch(r":?-+:?", cell)
                for cell in cells
            ):
                continue

            if len(cells) == 2:
                stripped = f"{cells[0]}: {cells[1]}"
            else:
                stripped = " - ".join(cells)

        stripped = re.sub(
            r"^[-*+]\s+",
            "- ",
            stripped
        )

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

    text = re.sub(
        r"\s+#{1,6}\s+",
        "\n\n",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def truncate_to_word_limit(text, max_words):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

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

    result = " ".join(selected).strip()

    result = re.sub(
        r"\n?\s*\d+\.\s*$",
        "",
        result
    ).strip()

    return result


# ============================================================
# ARGUMENTS
# ============================================================

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

parser.add_argument(
    "--workers",
    type=int,
    default=5,
    help="Maximum number of chunks processed in parallel"
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
print("Parallel workers:", args.workers)


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
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(index, chunk):
    chunk_number = index + 1

    print(
        f"\nStarting chunk {chunk_number}/{len(chunks)}..."
    )

    prompt = extract_template.render(
        source_text=chunk,
        max_sentences=args.max_sentences
    )

    result = None
    max_retries = 3
    calls = 0

    for attempt in range(
        1,
        max_retries + 1
    ):
        try:
            result = generate_summary(
                prompt
            )

            calls += 1

            if result is not None and result.strip():
                break

            print(
                f"Chunk {chunk_number}: empty response "
                f"(attempt {attempt}/{max_retries})"
            )

        except RuntimeError as e:
            calls += 1

            print(
                f"Chunk {chunk_number}: model error "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

        if attempt < max_retries:
            time.sleep(2)

    if result is None or not result.strip():
        raise RuntimeError(
            f"No model output for chunk {chunk_number} "
            f"after {max_retries} attempts"
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
        f"Chunk {chunk_number} complete — "
        f"Extracted: {len(extracted_sentences)}, "
        f"Validated: {len(validated_sentences)}"
    )

    return {
        "index": index,
        "extracted": extracted_sentences,
        "validated": validated_sentences,
        "model_calls": calls
    }


# ============================================================
# STATS
# ============================================================

start_time = time.time()

total_extracted = 0
total_validated = 0
model_calls = 0

all_validated_sentences = []


# ============================================================
# STEP 1: PARALLEL EXTRACTION + VALIDATION
# ============================================================

print(
    "\nStarting parallel key-sentence extraction..."
)

results = []

with ThreadPoolExecutor(
    max_workers=args.workers
) as executor:

    futures = {
        executor.submit(
            process_chunk,
            i,
            chunk
        ): i
        for i, chunk in enumerate(chunks)
    }

    for future in as_completed(futures):

        chunk_index = futures[future]

        try:
            result = future.result()
            results.append(result)

        except Exception as e:
            raise RuntimeError(
                f"Chunk {chunk_index + 1} failed: {e}"
            ) from e


# ============================================================
# RESTORE ORIGINAL DOCUMENT ORDER
# ============================================================

results.sort(
    key=lambda item: item["index"]
)

for result in results:

    total_extracted += len(
        result["extracted"]
    )

    total_validated += len(
        result["validated"]
    )

    model_calls += result[
        "model_calls"
    ]

    all_validated_sentences.extend(
        result["validated"]
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

final_summary = clean_plain_text(
    final_summary
)

print(
    "Initial final summary words:",
    count_words(final_summary)
)

# ============================================================
# STEP 3: LENGTH CONTROL
# ============================================================

current_words = count_words(
    final_summary
)

if current_words > args.summary_words:

    print(
        f"Summary is {current_words} words. "
        "Applying deterministic sentence-safe truncation."
    )

    final_summary = truncate_to_word_limit(
        final_summary,
        args.summary_words
    )

final_summary = clean_plain_text(
    final_summary
)


# ============================================================
# FINAL WORD-LIMIT FALLBACK
# ============================================================

if count_words(final_summary) > args.summary_words:

    print(
        "Applying final deterministic word-limit fallback..."
    )

    final_summary = truncate_to_word_limit(
        final_summary,
        args.summary_words
    )

final_summary = clean_plain_text(
    final_summary
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
    "Parallel workers:",
    args.workers
)

print(
    "Support saved:",
    support_path
)

print(
    "Summary saved:",
    summary_path
)