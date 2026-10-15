import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


def count_words(text):
    return len(text.split())


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

all_supported_claims = []

total_extracted = 0
supported = 0
partial = 0
unsupported = 0
contradicted = 0

model_calls = 0
start_time = time.time()

document_themes = []
themes_text = ""


# ============================================================
# OPTIONAL DOCUMENT-LEVEL THEME EXTRACTION
# ============================================================

if args.use_themes:

    sample_chunks = []

    if len(chunks) == 1:
        sample_chunks = [
            chunks[0]
        ]

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
# PROCESS ALL CHUNKS
# ============================================================

for i, chunk in enumerate(
    chunks,
    start=1
):

    print(
        f"\nProcessing chunk {i}/{len(chunks)}..."
    )


    # --------------------------------------------------------
    # EXTRACT CLAIMS
    # --------------------------------------------------------

    extract_prompt = extract_template.render(
        source_text=chunk,
        max_claims=args.max_claims,
        document_themes=themes_text,
        use_themes=args.use_themes
    )

    extract_result = generate_summary(
        extract_prompt
    )

    model_calls += 1

    if extract_result is None:
        print(
            "No extraction output."
        )
        continue

    try:
        extracted_json = json.loads(
            extract_result
        )

        claims = extracted_json.get(
            "claims",
            []
        )

    except json.JSONDecodeError:
        print(
            "Extraction output is not valid JSON."
        )
        continue

    total_extracted += len(claims)

    print(
        "Claims extracted:",
        len(claims)
    )


    # --------------------------------------------------------
    # VERIFY CLAIMS
    # --------------------------------------------------------

    if not claims:
        continue

    claims_text = "\n".join(
        claims
    )

    verify_prompt = verify_template.render(
        source_text=chunk,
        summary=claims_text
    )

    verify_result = generate_summary(
        verify_prompt
    )

    model_calls += 1

    if verify_result is None:
        print(
            "No verification output."
        )
        continue

    try:
        verified_json = json.loads(
            verify_result
        )

    except json.JSONDecodeError:
        print(
            "Verification output is not valid JSON."
        )
        continue

    verified_claims = verified_json.get(
        "claims",
        []
    )


    # --------------------------------------------------------
    # PROCESS LABELS
    # --------------------------------------------------------

    chunk_supported = 0

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
            chunk_supported += 1

            if claim_text:
                all_supported_claims.append(
                    claim_text
                )

        elif label == "partial":
            partial += 1

        elif label == "unsupported":
            unsupported += 1

        elif label == "contradicted":
            contradicted += 1

    print(
        "Supported:",
        chunk_supported
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

if final_summary is None:
    raise RuntimeError(
        "No final summary was generated."
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
    "Claims saved:",
    claims_path
)

print(
    "Summary saved:",
    summary_path
)