import argparse
import json
import re
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary
import unicodedata

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Verify a final summary against the original source document"
)

parser.add_argument(
    "source_file",
    help="Path to the original source document"
)

parser.add_argument(
    "summary_file",
    help="Path to the generated summary txt file"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=8000
)

parser.add_argument(
    "--output-file",
    required=True
)

parser.add_argument(
    "--only-chunks",
    type=str,
    default="",
    help="Optional comma-separated chunk numbers, e.g. 4,9,13,16"
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def extract_json(text):
    if text is None:
        return None

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            return None

    return None

def split_summary_into_statements(text):
    # Normalize Unicode characters and spaces
    text = unicodedata.normalize("NFKC", text)

    text = (
        text
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("**", "")
        .replace("__", "")
    )

    # Remove known headings
    headings = [
        "Major Themes of the Supported Claims – Compressed Summary (≈ 1 000 words)",
        "Major Themes of the Supported Claims - Compressed Summary (≈ 1 000 words)",
        "Origin and Legal Basis",
        "Scope, Definitions, and Exclusions",
        "Lawful Processing, Consent, and Derogations",
        "Data-Subject Rights",
        "Controller and Processor Obligations",
        "International Transfers and Adequacy",
        "Governance, Supervisory Authorities, and the European Data Protection Board (EDPB)",
        "Enforcement, Complaints, and Penalties",
    ]

    for heading in headings:
        text = text.replace(heading, " ")


    text = re.sub(
        r"(?<!\d)\b[1-9]\.\s+",
        " ",
        text
    )

    # Remove bullet markers
    text = re.sub(
        r"\s*[-•–—]\s+",
        " ",
        text
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Protect legal abbreviations before sentence splitting
    text = re.sub(
        r"\bArt\.\s*(\d+)",
        r"Art__DOT__\1",
        text
    )

    text = re.sub(
        r"\bArts\.\s*(\d+)",
        r"Arts__DOT__\1",
        text
    )

    text = text.replace(
        "e.g.",
        "e__DOT__g__DOT__"
    )

    text = text.replace(
        "i.e.",
        "i__DOT__e__DOT__"
    )

    # Split into factual sentences
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z“\"'])",
        text
    )

    statements = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Restore abbreviations
        part = re.sub(
            r"\bArt__DOT__(\d+)",
            r"Art. \1",
            part
        )

        part = re.sub(
            r"\bArts__DOT__(\d+)",
            r"Arts. \1",
            part
        )

        part = part.replace(
            "e__DOT__g__DOT__",
            "e.g."
        )

        part = part.replace(
            "i__DOT__e__DOT__",
            "i.e."
        )

        statements.append(part)

    return statements
def load_existing_results(output_path):
    if not output_path.exists():
        return {}

    try:
        data = json.loads(
            output_path.read_text(
                encoding="utf-8"
            )
        )

        existing = {}

        for item in data.get("results", []):
            existing[item["chunk"]] = item["result"]

        return existing

    except Exception:
        return {}


# ============================================================
# LOAD FILES
# ============================================================

source_text = load_document(
    args.source_file
)

summary_text = Path(
    args.summary_file
).read_text(
    encoding="utf-8"
)


# ============================================================
# PREPARE NUMBERED SUMMARY STATEMENTS
# ============================================================

statements = split_summary_into_statements(
    summary_text
)

numbered_summary = "\n".join(
    f"[{i}] {statement}"
    for i, statement in enumerate(
        statements,
        start=1
    )
)

print(
    "Summary statements:",
    len(statements)
)


# ============================================================
# SPLIT SOURCE
# ============================================================

chunks = chunk_document(
    source_text,
    max_tokens=args.context_size
)

print(
    "Source chunks:",
    len(chunks)
)


# ============================================================
# SELECT CHUNKS
# ============================================================

if args.only_chunks.strip():

    selected_chunks = [
        int(x.strip())
        for x in args.only_chunks.split(",")
        if x.strip()
    ]

else:

    selected_chunks = list(
        range(1, len(chunks) + 1)
    )


print(
    "Chunks to verify:",
    selected_chunks
)


# ============================================================
# LOAD PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

verify_template = env.get_template(
    "verify_final_summary_source.j2"
)


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

output_path = Path(
    args.output_file
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_by_chunk = load_existing_results(
    output_path
)


# ============================================================
# VERIFY
# ============================================================

max_retries = 3


for i in selected_chunks:

    if i < 1 or i > len(chunks):

        print(
            f"Skipping invalid chunk number: {i}"
        )

        continue

    chunk = chunks[i - 1]

    print(
        f"Verifying against source chunk {i}/{len(chunks)}..."
    )

    prompt = verify_template.render(
        source_text=chunk,
        summary_text=numbered_summary
    )

    parsed = None


    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            result = generate_summary(
                prompt
            )

        except RuntimeError as e:

            print(
                f"Chunk {i}: model error "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

            if attempt < max_retries:
                time.sleep(2)

            continue


        if result is None or not result.strip():

            print(
                f"Chunk {i}: empty response "
                f"(attempt {attempt}/{max_retries})"
            )

            if attempt < max_retries:
                time.sleep(2)

            continue


        parsed = extract_json(
            result
        )

        if parsed is not None:
            break


        print(
            f"Chunk {i}: invalid JSON "
            f"(attempt {attempt}/{max_retries})"
        )

        if attempt < max_retries:
            time.sleep(2)


    if parsed is None:

        print(
            f"Chunk {i}: FAILED after {max_retries} attempts"
        )

        continue


    results_by_chunk[i] = parsed

    print(
        f"Chunk {i}: success"
    )


# ============================================================
# BUILD COMPLETE RESULT
# ============================================================

all_results = [
    {
        "chunk": chunk_number,
        "result": results_by_chunk[chunk_number]
    }
    for chunk_number in sorted(results_by_chunk)
]


missing_chunks = [
    i
    for i in range(1, len(chunks) + 1)
    if i not in results_by_chunk
]

# ============================================================
# DOCUMENT-LEVEL AGGREGATION
# ============================================================

statement_aggregation = {}

for statement_id, statement in enumerate(
    statements,
    start=1
):

    labels = []
    supporting_chunks = []
    partial_chunks = []
    contradicted_chunks = []

    for chunk_number, chunk_result in results_by_chunk.items():

        for item in chunk_result.get(
            "statements",
            []
        ):

            if item.get("id") != statement_id:
                continue

            label = item.get(
                "label",
                "unsupported"
            )

            labels.append(label)

            if label == "supported":
                supporting_chunks.append(
                    chunk_number
                )

            elif label == "partial":
                partial_chunks.append(
                    chunk_number
                )

            elif label == "contradicted":
                contradicted_chunks.append(
                    chunk_number
                )

    # Final document-level status:
    # support anywhere in the document counts as support.
    if "supported" in labels:
        final_status = "supported"

    elif "partial" in labels:
        final_status = "partial"

    elif "contradicted" in labels:
        final_status = "contradicted"

    else:
        final_status = "unsupported"

    statement_aggregation[
        str(statement_id)
    ] = {
        "statement": statement,
        "final_status": final_status,
        "supporting_chunks": supporting_chunks,
        "partial_chunks": partial_chunks,
        "contradicted_chunks": contradicted_chunks
    }


# ============================================================
# AGGREGATE COUNTS
# ============================================================

status_counts = {
    "supported": 0,
    "partial": 0,
    "unsupported": 0,
    "contradicted": 0
}

for item in statement_aggregation.values():

    status = item[
        "final_status"
    ]

    status_counts[
        status
    ] += 1


total_statements = len(
    statement_aggregation
)

supported_statements = status_counts[
    "supported"
]

support_percentage = (
    supported_statements
    / total_statements
    * 100
    if total_statements
    else 0
)

output_data = {
    "total_chunks": len(chunks),

    "summary_statements": {
        str(i): statement
        for i, statement in enumerate(
            statements,
            start=1
        )
    },

    "successful_chunks": len(all_results),
    "failed_chunks": missing_chunks,

    "document_level_verification": {
        "total_statements": total_statements,
        "supported": status_counts["supported"],
        "partial": status_counts["partial"],
        "unsupported": status_counts["unsupported"],
        "contradicted": status_counts["contradicted"],
        "support_percentage": round(
            support_percentage,
            2
        )
    },

    "statement_aggregation": statement_aggregation,

    "results": all_results
}


# ============================================================
# SAVE
# ============================================================

output_path.write_text(
    json.dumps(
        output_data,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# RESULT
# ============================================================

print()
print("Verification complete.")

print(
    "Successful chunks:",
    len(all_results),
    "/",
    len(chunks)
)

if missing_chunks:

    print(
        "Still missing chunks:",
        missing_chunks
    )

else:

    print(
        "All chunks verified successfully."
    )

print()
print("=" * 60)
print("DOCUMENT-LEVEL VERIFICATION")
print("=" * 60)

print(
    "Total statements:",
    total_statements
)

print(
    "Supported:",
    status_counts["supported"]
)

print(
    "Partial:",
    status_counts["partial"]
)

print(
    "Unsupported:",
    status_counts["unsupported"]
)

print(
    "Contradicted:",
    status_counts["contradicted"]
)

print(
    "Support percentage:",
    f"{support_percentage:.2f}%"
)

print(
    "Saved:",
    output_path
)