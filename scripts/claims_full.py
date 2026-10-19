import argparse
import json
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
    description="Extract, verify, and summarize claims from a full document"
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
    "--max-claims",
    type=int,
    default=10,
    help="Maximum claims per chunk"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=2000,
    help="Target final summary length in words"
)

parser.add_argument(
    "--additional-instructions",
    type=str,
    default="",
    help="Optional extra instructions for summary focus"
)

parser.add_argument(
    "--output-name",
    required=True,
    help="Output name, e.g. ai_act or gdpr"
)

parser.add_argument(
    "--use-themes",
    action="store_true",
    help="Enable experimental document-level theme guidance"
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

document_text = load_document(args.file)

chunks = chunk_document(
    document_text,
    max_tokens=args.context_size
)

print("Total chunks:", len(chunks))
print("Context size:", args.context_size)
print("Target summary:", args.summary_words, "words")
print("Parallel workers:", args.workers)
print(
    "Theme guidance:",
    "ENABLED" if args.use_themes else "DISABLED"
)


# ============================================================
# LOAD PROMPTS
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

extract_template = env.get_template(
    "extract_claims.j2"
)

verify_template = env.get_template(
    "verify_claims.j2"
)

summary_template = env.get_template(
    "summarize_from_claims.j2"
)

compress_template = env.get_template(
    "compress.j2"
)

themes_template = None

if args.use_themes:
    themes_template = env.get_template(
        "extract_document_themes.j2"
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

start_time = time.time()

document_themes = []
themes_text = ""

model_calls = 0


# ============================================================
# OPTIONAL DOCUMENT-LEVEL THEME EXTRACTION
# ============================================================

if args.use_themes:

    if len(chunks) == 1:
        sample_chunks = [chunks[0]]

    elif len(chunks) == 2:
        sample_chunks = [
            chunks[0],
            chunks[1]
        ]

    else:
        middle_index = len(chunks) // 2

        sample_chunks = [
            chunks[0],
            chunks[middle_index],
            chunks[-1]
        ]

    document_sample = "\n\n".join(
        sample_chunks
    )

    themes_prompt = themes_template.render(
        document_sample=document_sample,
        max_themes=8
    )

    print(
        "\nExtracting document-level themes..."
    )

    themes_result = generate_summary(
        themes_prompt
    )

    model_calls += 1

    if themes_result:
        try:
            themes_json = json.loads(
                themes_result
            )

            document_themes = themes_json.get(
                "themes",
                []
            )

        except json.JSONDecodeError:
            print(
                "Theme extraction output is not valid JSON."
            )

    print(
        "Document themes:",
        document_themes
    )

    themes_text = "\n".join(
        f"- {theme}"
        for theme in document_themes
    )


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(index, chunk):
    chunk_number = index + 1
    calls = 0

    print(
        f"\nStarting chunk {chunk_number}/{len(chunks)}..."
    )

    # EXTRACT CLAIMS

    extract_prompt = extract_template.render(
        source_text=chunk,
        max_claims=args.max_claims,
        document_themes=themes_text,
        use_themes=args.use_themes
    )

    extract_result = generate_summary(
        extract_prompt
    )

    calls += 1

    if extract_result is None:
        raise RuntimeError(
            f"No extraction output for chunk {chunk_number}"
        )

    try:
        extracted_json = json.loads(
            extract_result
        )

        claims = extracted_json.get(
            "claims",
            []
        )

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid extraction JSON for chunk {chunk_number}"
        ) from e

    print(
        f"Chunk {chunk_number}: extracted {len(claims)} claims"
    )

    if not claims:
        return {
            "index": index,
            "claims": [],
            "supported_claims": [],
            "supported": 0,
            "partial": 0,
            "unsupported": 0,
            "contradicted": 0,
            "model_calls": calls
        }

    # VERIFY CLAIMS

    claims_text = "\n".join(
        claims
    )

    verify_prompt = verify_template.render(
        source_text=chunk,
        summary=claims_text
    )

    verified_json = None
    last_error = None

    for attempt in range(2):

        verify_result = generate_summary(
            verify_prompt
        )

        calls += 1

        if verify_result is None:
            last_error = RuntimeError(
                f"No verification output for chunk {chunk_number}"
            )
            continue

        try:
            verified_json = json.loads(
                verify_result
            )
            break

        except json.JSONDecodeError as e:
            last_error = e

            print(
                f"Chunk {chunk_number}: invalid verification JSON "
                f"(attempt {attempt + 1}/2)"
            )

    if verified_json is None:
        raise RuntimeError(
            f"Invalid verification JSON for chunk {chunk_number} "
            "after 2 attempts"
        ) from last_error


    verified_claims = verified_json.get(
        "claims",
        []
    )

    supported_claims = []

    supported = 0
    partial = 0
    unsupported = 0
    contradicted = 0

    for claim in verified_claims:

        label = claim.get(
            "label"
        )

        claim_text = claim.get(
            "claim",
            ""
        )

        if label == "supported":
            supported += 1

            if claim_text:
                supported_claims.append(
                    claim_text
                )

        elif label == "partial":
            partial += 1

        elif label == "unsupported":
            unsupported += 1

        elif label == "contradicted":
            contradicted += 1

    print(
        f"Chunk {chunk_number} complete — "
        f"Supported: {supported}, "
        f"Partial: {partial}, "
        f"Unsupported: {unsupported}, "
        f"Contradicted: {contradicted}"
    )

    return {
        "index": index,
        "claims": claims,
        "supported_claims": supported_claims,
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "model_calls": calls
    }


# ============================================================
# PARALLEL CHUNK PROCESSING
# ============================================================

print(
    "\nStarting parallel Claims extraction + verification..."
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

all_supported_claims = []

total_extracted = 0
supported = 0
partial = 0
unsupported = 0
contradicted = 0

for result in results:

    total_extracted += len(
        result["claims"]
    )

    supported += result["supported"]
    partial += result["partial"]
    unsupported += result["unsupported"]
    contradicted += result["contradicted"]

    model_calls += result[
        "model_calls"
    ]

    all_supported_claims.extend(
        result["supported_claims"]
    )


# ============================================================
# SAVE SUPPORTED CLAIMS
# ============================================================

claims_path = os.path.join(
    output_dir,
    f"{args.output_name}_supported_claims.txt"
)

with open(
    claims_path,
    "w",
    encoding="utf-8"
) as f:

    for claim in all_supported_claims:
        f.write(
            claim + "\n"
        )


# ============================================================
# GENERATE FINAL SUMMARY
# ============================================================

claims_text = "\n".join(
    all_supported_claims
)

print(
    "\nGenerating final summary from supported claims..."
)

summary_prompt = summary_template.render(
    claims_text=claims_text,
    document_themes=themes_text,
    use_themes=args.use_themes,
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
# COMPRESS IF TOO LONG
# ============================================================

compression_calls = 0

current_words = count_words(
    final_summary
)

if current_words > args.summary_words:

    if current_words <= int(
        args.summary_words * 1.10
    ):
        print(
            f"Small overshoot ({current_words} words). "
            "Using deterministic sentence-safe truncation."
        )

        final_summary = truncate_to_word_limit(
            final_summary,
            args.summary_words
        )

    else:
        print(
            f"Large overshoot ({current_words} words). "
            "Using one LLM compression call."
        )

        compress_prompt = compress_template.render(
            summary=final_summary,
            summary_length=args.summary_words
        )

        compressed = generate_summary(
            compress_prompt
        )

        model_calls += 1
        compression_calls += 1

        if compressed is not None and compressed.strip():

            final_summary = clean_plain_text(
                compressed
            )

            print(
                "Words after compression:",
                count_words(final_summary)
            )

        else:
            print(
                "Compression returned no text."
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
    f"{args.output_name}_claims_summary_{args.summary_words}_words.txt"
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

print(
    "\n" + "=" * 80
)

print(
    "CLAIM-BASED SUMMARIZATION COMPLETE"
)

print(
    "=" * 80
)

print(
    "Mode:",
    "THEME-GUIDED EXPERIMENT"
    if args.use_themes
    else "BASELINE"
)

print(
    "Total extracted claims:",
    total_extracted
)

print(
    "Supported:",
    supported
)

print(
    "Partial:",
    partial
)

print(
    "Unsupported:",
    unsupported
)

print(
    "Contradicted:",
    contradicted
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
    "Compression calls:",
    compression_calls
)
print(
    "Parallel workers:",
    args.workers
)

print(
    "Claims saved:",
    claims_path
)

print(
    "Summary saved:",
    summary_path
)