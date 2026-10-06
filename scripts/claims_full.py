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
    description="Extract and verify claims from a full document"
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


    # ========================================================
    # EXTRACT CLAIMS
    # ========================================================

    extract_prompt = extract_template.render(
        source_text=chunk,
        max_claims=args.max_claims
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


    # ========================================================
    # VERIFY CLAIMS
    # ========================================================

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


    # ========================================================
    # PROCESS LABELS
    # ========================================================

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
# SAVE FINAL SUPPORTED CLAIMS
# ============================================================

output_path = os.path.join(
    output_dir,
    f"{args.output_name}_claims.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    for claim in all_supported_claims:
        f.write(
            claim + "\n"
        )


# ============================================================
# FINAL STATS
# ============================================================

runtime = time.time() - start_time

print("\n" + "=" * 80)
print("CLAIM EXTRACTION COMPLETE")
print("=" * 80)

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