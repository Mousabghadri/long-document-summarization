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
from src.llm import generate_summary
from src.postprocess import count_words, finalize_summary, prompt_env
from src.run_metadata import write_metadata, print_run_summary


_QUOTES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00ad": None
})


def normalize_text(text):
    """
    Key used for exact-copy validation.  All whitespace is removed and quote /
    dash variants are unified, so a sentence only passes if its characters
    appear verbatim in the chunk, but line breaks, hyphenation spaces or PDF
    extraction artefacts inside the source cannot make a correct copy fail.
    """

    text = unicodedata.normalize(
        "NFKC",
        text
    ).translate(_QUOTES)

    return re.sub(r"\s+", "", text)


def strip_line_markers(line):
    """Remove list markers / quotes the model may add despite instructions."""

    line = line.strip()

    line = re.sub(
        r"^(?:[-*•]\s+|\d+[.)]\s+)",
        "",
        line
    )

    if len(line) > 1 and line[0] in "\"'“" and line[-1] in "\"'”":
        line = line[1:-1]

    return line.strip()


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
    default=1000,
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
    help="Optional extra instructions (do NOT use in the fair comparison)"
)

parser.add_argument(
    "--workers",
    type=int,
    default=5,
    help="Maximum number of chunks processed in parallel"
)

parser.add_argument(
    "--summary-prompt",
    default="summarize_from_support.j2",
    help="Prompt template for the final summary step"
)

parser.add_argument(
    "--support-file",
    default=None,
    help="Reuse support sentences saved by an earlier run (skips extraction; "
         "for prompt experiments, so only the final step differs)"
)

args = parser.parse_args()


document_text = load_document(args.file)

chunks = chunk_document(
    document_text,
    max_tokens=args.context_size
)

if not chunks:
    raise ValueError("No chunks were created from the document.")


print("Total chunks:", len(chunks))
print("Context size:", args.context_size)
print("Target summary:", args.summary_words, "words")
print("Parallel workers:", args.workers)


env = prompt_env()

extract_template = env.get_template("extract_key_sentences.j2")
summary_template = env.get_template(args.summary_prompt)


output_dir = os.path.join(
    "outputs",
    "method_outputs",
    args.output_name
)

os.makedirs(
    output_dir,
    exist_ok=True
)


def process_chunk(index, chunk):

    chunk_number = index + 1

    print(
        f"\nStarting chunk {chunk_number}/{len(chunks)}..."
    )

    try:

        result = generate_summary(
            extract_template.render(
                source_text=chunk,
                max_sentences=args.max_sentences
            ),
            purpose="key_sentences_extract"
        )

    except RuntimeError as e:

        # A chunk with nothing extractable (tariff tables, annex lists ...) can
        # make the model return no text; skip that chunk instead of losing the
        # whole document.  Other errors still fail the run.

        if "no text content" not in str(e):
            raise

        print(
            f"Chunk {chunk_number}: model returned no sentences - chunk skipped"
        )

        return {
            "index": index,
            "extracted": [],
            "validated": [],
            "skipped": True
        }

    extracted = [
        strip_line_markers(l)
        for l in result.splitlines()
        if l.strip()
    ]

    extracted = [
        l
        for l in extracted
        if l
    ]

    normalized_chunk = normalize_text(chunk)

    validated, seen = [], set()

    for sentence in extracted:

        norm = normalize_text(sentence)

        if len(sentence.split()) >= 3 and norm in normalized_chunk and norm not in seen:

            validated.append(sentence)

            seen.add(norm)

    # the budget is enforced deterministically, so no method can exceed it

    validated = validated[:args.max_sentences]

    print(
        f"Chunk {chunk_number} complete - "
        f"Extracted: {len(extracted)}, "
        f"Validated: {len(validated)}"
    )

    return {
        "index": index,
        "extracted": extracted,
        "validated": validated
    }


start_time = time.time()


if args.support_file:

    reused = [
        l.strip()
        for l in open(args.support_file, encoding="utf-8")
        if l.strip()
    ]

    print(
        f"\nReusing {len(reused)} support sentences "
        f"from {args.support_file} (no extraction)"
    )

    results = [
        {
            "index": 0,
            "extracted": reused,
            "validated": reused
        }
    ]

else:

    print("\nStarting parallel key-sentence extraction...")

    results = []


if not args.support_file:

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(process_chunk, i, c): i
            for i, c in enumerate(chunks)
        }

        for future in as_completed(futures):

            i = futures[future]

            try:
                results.append(future.result())

            except Exception as e:
                raise RuntimeError(
                    f"Chunk {i + 1} failed: {e}"
                ) from e


results.sort(
    key=lambda r: r["index"]
)

total_extracted = sum(
    len(r["extracted"])
    for r in results
)

all_validated = [
    s
    for r in results
    for s in r["validated"]
]


support_path = os.path.join(
    output_dir,
    f"{args.output_name}_support_sentences.txt"
)

with open(support_path, "w", encoding="utf-8") as f:

    for sentence in all_validated:
        f.write(sentence + "\n")


if not all_validated:
    raise RuntimeError("No validated support sentences - cannot build a summary.")


print("\nGenerating final summary from validated support...")


draft = generate_summary(
    summary_template.render(
        support_text="\n".join(all_validated),
        summary_length=args.summary_words,
        additional_instructions=args.additional_instructions
    ),
    purpose="final_summary"
)


final_summary, final_info = finalize_summary(
    draft,
    args.summary_words,
    compress_call=lambda p: generate_summary(
        p,
        purpose="final_compress"
    )
)

final_words = count_words(final_summary)


summary_path = os.path.join(
    output_dir,
    f"{args.output_name}_key_sentences_summary_{args.summary_words}_words.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(final_summary)


validation_rate = (
    len(all_validated) / total_extracted * 100
    if total_extracted
    else 0
)


meta_path = write_metadata(
    output_dir,
    "key_sentences",
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
        "extracted_sentences": total_extracted,
        "validated_sentences": len(all_validated),
        "validation_rate": round(validation_rate, 2),
        "support_words": count_words("\n".join(all_validated)),
        "final_length_control": final_info,
        "additional_instructions": args.additional_instructions or None,
        "summary_prompt": args.summary_prompt,
        "skipped_chunks": sum(
            1 for r in results
            if r.get("skipped")
        ),
        "support_reused_from": args.support_file,
    }
)


print_run_summary(
    "extract-support (key sentences)",
    final_words,
    start_time
)

print(
    "Validated support sentences:",
    len(all_validated),
    f"({validation_rate:.2f}%)"
)

print("Support saved:", support_path)
print("Summary saved:", summary_path)
print("Metadata saved:", meta_path)