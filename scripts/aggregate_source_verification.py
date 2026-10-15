import argparse
import json
from pathlib import Path


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Aggregate chunk-level source verification results."
)

parser.add_argument(
    "verification_file",
    help="Path to source verification JSON file"
)

parser.add_argument(
    "--output-file",
    required=True,
    help="Path to save aggregated JSON"
)

args = parser.parse_args()


# ============================================================
# LOAD VERIFICATION FILE
# ============================================================

input_path = Path(
    args.verification_file
)

data = json.loads(
    input_path.read_text(
        encoding="utf-8"
    )
)

summary_statements = data.get(
    "summary_statements",
    {}
)

results = data.get(
    "results",
    []
)


# ============================================================
# BUILD CHUNK LOOKUP
# ============================================================

results_by_chunk = {}

for item in results:

    chunk_number = item.get(
        "chunk"
    )

    chunk_result = item.get(
        "result",
        {}
    )

    results_by_chunk[
        chunk_number
    ] = chunk_result


# ============================================================
# AGGREGATE EACH STATEMENT
# ============================================================

statement_aggregation = {}

for statement_id_str, statement in summary_statements.items():

    statement_id = int(
        statement_id_str
    )

    labels = []

    supporting_chunks = []
    partial_chunks = []
    contradicted_chunks = []
    unsupported_chunks = []

    for chunk_number, chunk_result in results_by_chunk.items():

        chunk_statements = chunk_result.get(
            "statements",
            []
        )

        for item in chunk_statements:

            if item.get("id") != statement_id:
                continue

            label = item.get(
                "label",
                "unsupported"
            )

            labels.append(
                label
            )

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

            elif label == "unsupported":

                unsupported_chunks.append(
                    chunk_number
                )

    # ========================================================
    # DOCUMENT-LEVEL DECISION
    # ========================================================

    if "supported" in labels:

        final_status = "supported"

    elif "partial" in labels:

        final_status = "partial"

    elif "contradicted" in labels:

        final_status = "contradicted"

    else:

        final_status = "unsupported"

    statement_aggregation[
        statement_id_str
    ] = {
        "statement": statement,
        "final_status": final_status,
        "supporting_chunks": supporting_chunks,
        "partial_chunks": partial_chunks,
        "contradicted_chunks": contradicted_chunks,
        "unsupported_chunks": unsupported_chunks
    }


# ============================================================
# COUNTS
# ============================================================

status_counts = {
    "supported": 0,
    "partial": 0,
    "unsupported": 0,
    "contradicted": 0
}

for item in statement_aggregation.values():

    status = item.get(
        "final_status"
    )

    if status in status_counts:

        status_counts[
            status
        ] += 1


total_statements = len(
    statement_aggregation
)

support_percentage = (
    status_counts["supported"]
    / total_statements
    * 100
    if total_statements
    else 0
)


# ============================================================
# OUTPUT DATA
# ============================================================

output_data = {
    "source_verification_file": str(
        input_path
    ),

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

    "statement_aggregation": statement_aggregation
}


# ============================================================
# SAVE
# ============================================================

output_path = Path(
    args.output_file
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    json.dumps(
        output_data,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT RESULT
# ============================================================

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

print()
print(
    "Saved:",
    output_path
)