

import argparse
import os
import re
import sys
import unicodedata
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary, generate_json
from src.postprocess import count_words, finalize_summary, prompt_env
from src.run_metadata import write_metadata, print_run_summary


_QUOTES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00ad": None,
})


def normalize_text(text):
    """
    Key used for exact-copy validation.

    All whitespace is removed and quote/dash variants
    are unified, so a sentence only passes if its
    characters appear verbatim in the chunk.
    """

    text = unicodedata.normalize("NFKC", text)

    text = text.translate(_QUOTES)

    return re.sub(r"\s+", "", text)


def strip_line_markers(line):
    """
    Remove list markers or quotes that the model
    may add despite instructions.
    """

    line = line.strip()

    line = re.sub(
        r"^(?:[-*•]\s+|\d+[.)]\s+)",
        "",
        line,
    )

    if (
        len(line) > 1
        and line[0] in "\"'“"
        and line[-1] in "\"'”"
    ):
        line = line[1:-1]

    return line.strip()


parser = argparse.ArgumentParser(
    description=(
        "Extract validated support sentences "
        "and generate a grounded summary"
    )
)

parser.add_argument(
    "file",
    help="Path to the document file",
)

parser.add_argument(
    "--context-size",
    type=int,
    default=8000,
    help="Maximum source tokens per chunk",
)

parser.add_argument(
    "--max-sentences",
    type=int,
    default=10,
    help="Final budget per chunk: total selected = max_sentences x number of chunks"
)
parser.add_argument(
    "--candidates-per-chunk",
    type=int,
    default=20,
    help="Candidate sentences extracted from each chunk before global selection"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=1000,
    help="Target final summary length in words",
)

parser.add_argument(
    "--output-name",
    required=True,
    help="Output name, e.g. ai_act or gdpr",
)

parser.add_argument(
    "--additional-instructions",
    type=str,
    default="",
    help="Optional extra instructions (do NOT use in the fair comparison)",
)

parser.add_argument(
    "--workers",
    type=int,
    default=5,
    help="Maximum number of chunks processed in parallel",
)

args = parser.parse_args()


document_text = load_document(args.file)

chunks = chunk_document(
    document_text,
    max_tokens=args.context_size,
)

if not chunks:
    raise ValueError(
        "No chunks were created from the document."
    )


print("Total chunks:", len(chunks))
print("Context size:", args.context_size)
print("Target summary:", args.summary_words, "words")
print("Parallel workers:", args.workers)


env = prompt_env()

extract_template = env.get_template(
    "extract_key_sentences.j2"
)

summary_template = env.get_template(
    "summarize_from_support.j2"
)

select_template = env.get_template(
    "select_global_sentences.j2"
)


output_dir = os.path.join(
    "outputs",
    "method_outputs",
    args.output_name,
)

os.makedirs(
    output_dir,
    exist_ok=True,
)


def process_chunk(index, chunk):
    chunk_number = index + 1
    print(f"\nStarting chunk {chunk_number}/{len(chunks)}...")

    prompt = extract_template.render(
        source_text=chunk,
        max_sentences=args.candidates_per_chunk
    )

    result = generate_summary(
        prompt,
        purpose="key_sentences_extract"
    )

    extracted = []
    for line in result.splitlines():
        if line.strip():
            cleaned = strip_line_markers(line)
            if cleaned:
                extracted.append(cleaned)

    normalized_chunk = normalize_text(chunk)

    validated = []
    seen = set()
    for sentence in extracted:
        norm = normalize_text(sentence)
        long_enough = len(sentence.split()) >= 3
        in_source = norm in normalized_chunk
        not_duplicate = norm not in seen
        if long_enough and in_source and not_duplicate:
            validated.append(sentence)
            seen.add(norm)

    validated = validated[:args.candidates_per_chunk]

    print(
        f"Chunk {chunk_number} complete - "
        f"Candidates extracted: {len(extracted)}, "
        f"Validated: {len(validated)}"
    )

    return {
        "index": index,
        "extracted": extracted,
        "validated": validated
    }

start_time = time.time()


print("\nStarting parallel key-sentence extraction...")

results = []

with ThreadPoolExecutor(
    max_workers=args.workers
) as executor:

    futures = {
        executor.submit(process_chunk, i, chunk): i
        for i, chunk in enumerate(chunks)
    }

    for future in as_completed(futures):

        i = futures[future]

        try:
            results.append(
                future.result()
            )

        except Exception as e:
            raise RuntimeError(
                f"Chunk {i + 1} failed: {e}"
            ) from e


results.sort(
    key=lambda result: result["index"]
)


total_extracted = sum(
    len(result["extracted"])
    for result in results
)

all_validated = [
    sentence
    for result in results
    for sentence in result["validated"]
]

# ============================================================
# GLOBAL SELECTION
# ============================================================

budget = args.max_sentences * len(chunks)

candidates = []
for r in results:
    part = r["index"] + 1
    for sentence in r["validated"]:
        candidates.append({
            "id": len(candidates) + 1,
            "part": part,
            "text": sentence
        })

selection_used = False
selection_fallback = False

if len(candidates) <= budget:
    print(
        f"\nOnly {len(candidates)} candidates for a budget of {budget}: "
        "keeping all of them."
    )
    selected = [c["text"] for c in candidates]

else:
    print(
        f"\nGlobal selection: {len(candidates)} candidates -> {budget} sentences"
    )

    candidates_text = "\n".join(
        f"[{c['id']}] (part {c['part']} of {len(chunks)}) {c['text']}"
        for c in candidates
    )

    select_prompt = select_template.render(
        candidates=candidates_text,
        budget=budget
    )

    try:
        answer = generate_json(
            select_prompt,
            purpose="key_sentences_global_select"
        )
        raw_ids = answer.get("selected_ids", [])
    except RuntimeError as e:
        print("Global selection failed:", e)
        raw_ids = []

    valid_ids = []
    seen_ids = set()
    for x in raw_ids:
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= len(candidates) and x not in seen_ids:
            valid_ids.append(x)
            seen_ids.add(x)

    if valid_ids:
        selection_used = True
        kept = valid_ids[:budget]
        ordered = sorted(kept)
        selected = [candidates[i - 1]["text"] for i in ordered]  
    else:
        selection_fallback = True
        print("No valid IDs returned - falling back to the first sentences of each chunk.")
        selected = []
        for r in results:
            selected.extend(r["validated"][:args.max_sentences])

selected_set = set(selected)

selected_per_part = []
for r in results:
    count = 0
    for sentence in r["validated"]:
        if sentence in selected_set:
            count += 1
    selected_per_part.append(count)

print("Selected sentences:", len(selected))
print("Selected per part:", selected_per_part)



support_path = os.path.join(
    output_dir,
    f"{args.output_name}_support_sentences.txt",
)

with open(
    support_path,
    "w",
    encoding="utf-8",
) as f:

    for sentence in selected:
        f.write(sentence + "\n")


if not selected:
    raise RuntimeError(
        "No validated support sentences - "
        "cannot build a summary."
    )


print(
    "\nGenerating final summary from validated support..."
)


draft = generate_summary(
    summary_template.render(
        support_text="\n".join(selected),
        summary_length=args.summary_words,
        additional_instructions=args.additional_instructions,
    ),
    purpose="final_summary",
)


final_summary, final_info = finalize_summary(
    draft,
    args.summary_words,
    compress_call=lambda prompt: generate_summary(
        prompt,
        purpose="final_compress",
    ),
)


final_words = count_words(final_summary)


summary_path = os.path.join(
    output_dir,
    (
        f"{args.output_name}_key_sentences_global_summary_"
        f"{args.summary_words}_words.txt"
    ),
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(final_summary)


validation_rate = (
    len(all_validated) / total_extracted * 100
    if total_extracted
    else 0
)


meta_path = write_metadata(
    output_dir,
    "key_sentences_global",
    args.file,
    document_text,
    start_time,
    args.summary_words,
    final_words,
    len(chunks),
    args.context_size,
    extra={
        "workers": args.workers,
        "max_sentences_per_chunk": args.max_sentences,
        "candidates_per_chunk": args.candidates_per_chunk,
        "extracted_sentences": total_extracted,
        "validated_candidates": len(all_validated),
        "validation_rate": round(validation_rate, 2),
        "selection_budget": budget,
        "selected_sentences": len(selected),
        "selected_per_part": selected_per_part,
        "selection_used": selection_used,
        "selection_fallback": selection_fallback,
        "support_words": count_words(
            "\n".join(selected)
        ),
        "final_length_control": final_info,
        "additional_instructions": (
            args.additional_instructions or None
        ),
    },)


print_run_summary(
    "extract-support (key sentences)",
    final_words,
    start_time,
)

print(
    "Validated support sentences:",
    len(all_validated),
    f"({validation_rate:.2f}%)",
)

print("Support saved:", support_path)
print("Summary saved:", summary_path)
print("Metadata saved:", meta_path)
