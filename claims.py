import argparse
import json
import os

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Extract and verify claims from a long document"
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
    "--max-claims",
    type=int,
    default=10,
    help="Maximum number of claims to extract per chunk"
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
# LOAD PROMPTS
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

extract_claims_template = env.get_template(
    "extract_claims.j2"
)

verify_claims_template = env.get_template(
    "verify_claims.j2"
)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

base_output_dir = "outputs/claims"

extracted_dir = os.path.join(
    base_output_dir,
    "extracted"
)

verified_dir = os.path.join(
    base_output_dir,
    "verified"
)

raw_dir = os.path.join(
    base_output_dir,
    "raw"
)

os.makedirs(
    extracted_dir,
    exist_ok=True
)

os.makedirs(
    verified_dir,
    exist_ok=True
)

os.makedirs(
    raw_dir,
    exist_ok=True
)


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


    # ========================================================
    # STEP 1: EXTRACT CLAIMS
    # ========================================================

    extract_prompt = extract_claims_template.render(
        source_text=chunk,
        max_claims=args.max_claims
    )

    extract_result = generate_summary(
        extract_prompt
    )


    # Save raw extraction output

    raw_extract_path = os.path.join(
        raw_dir,
        f"chunk_{i}_extraction.txt"
    )

    with open(
        raw_extract_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            extract_result
        )


    # Parse extracted claims

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


    print(
        "Claims extracted:",
        len(claims)
    )


    # Save extracted claims

    extracted_path = os.path.join(
        extracted_dir,
        f"chunk_{i}.json"
    )

    with open(
        extracted_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            extracted_json,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # STEP 2: VERIFY CLAIMS
    # ========================================================

    claims_text = "\n".join(
        claims
    )

    verify_prompt = verify_claims_template.render(
        source_text=chunk,
        summary=claims_text
    )

    verify_result = generate_summary(
        verify_prompt
    )


    # Save raw verification output

    raw_verify_path = os.path.join(
        raw_dir,
        f"chunk_{i}_verification.txt"
    )

    with open(
        raw_verify_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            verify_result
        )


    # Parse verification JSON

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
    # COUNT LABELS
    # ========================================================

    supported = 0
    partial = 0
    unsupported = 0
    contradicted = 0

    for claim in verified_claims:

        label = claim.get(
            "label"
        )

        if label == "supported":
            supported += 1

        elif label == "partial":
            partial += 1

        elif label == "unsupported":
            unsupported += 1

        elif label == "contradicted":
            contradicted += 1


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


    # ========================================================
    # SAVE VERIFIED CLAIMS
    # ========================================================

    verified_path = os.path.join(
        verified_dir,
        f"chunk_{i}.json"
    )

    with open(
        verified_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            verified_json,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "Saved extracted and verified claims for chunk",
        i
    )