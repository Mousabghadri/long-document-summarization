import json
import os


# ============================================================
# KEY SENTENCES EVALUATION
# ============================================================

key_base = "outputs/key_sentences"

extracted_dir = os.path.join(
    key_base,
    "extracted"
)

validated_dir = os.path.join(
    key_base,
    "validated"
)

rejected_dir = os.path.join(
    key_base,
    "rejected"
)


def count_lines(file_path):

    if not os.path.exists(file_path):
        return 0

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return len(
            [
                line
                for line in f
                if line.strip()
            ]
        )


total_extracted = 0
total_validated = 0
total_rejected = 0


if os.path.exists(extracted_dir):

    files = sorted(
        os.listdir(extracted_dir)
    )

    for filename in files:

        extracted_path = os.path.join(
            extracted_dir,
            filename
        )

        validated_path = os.path.join(
            validated_dir,
            filename
        )

        rejected_path = os.path.join(
            rejected_dir,
            filename
        )

        total_extracted += count_lines(
            extracted_path
        )

        total_validated += count_lines(
            validated_path
        )

        total_rejected += count_lines(
            rejected_path
        )


if total_extracted > 0:

    exact_match_rate = (
        total_validated
        / total_extracted
        * 100
    )

else:

    exact_match_rate = 0


# ============================================================
# CLAIMS EVALUATION
# ============================================================

claims_dir = "outputs/claims/verified"

total_claims = 0

supported = 0
partial = 0
unsupported = 0
contradicted = 0


if os.path.exists(claims_dir):

    files = sorted(
        os.listdir(claims_dir)
    )

    for filename in files:

        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(
            claims_dir,
            filename
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        claims = data.get(
            "claims",
            []
        )

        total_claims += len(
            claims
        )

        for claim in claims:

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


if total_claims > 0:

    supported_rate = (
        supported
        / total_claims
        * 100
    )

else:

    supported_rate = 0


# ============================================================
# SEQUENTIAL EVALUATION
# ============================================================

sequential_path = os.path.join(
    "outputs",
    "sequential",
    "result.json"
)

sequential_data = None

if os.path.exists(
    sequential_path
):

    with open(
        sequential_path,
        "r",
        encoding="utf-8"
    ) as f:

        sequential_data = json.load(f)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SEQUENTIAL EVALUATION")
print("=" * 60)

if sequential_data:

    print(
        "Document total chunks:",
        sequential_data.get(
            "document_total_chunks"
        )
    )

    print(
        "Processed chunks:",
        sequential_data.get(
            "processed_chunks"
        )
    )

    print(
        "Context size:",
        sequential_data.get(
            "context_size"
        )
    )

    print(
        "Target summary tokens:",
        sequential_data.get(
            "summary_size_target"
        )
    )

    print(
        "Final summary tokens:",
        sequential_data.get(
            "final_summary_tokens"
        )
    )

    target = sequential_data.get(
        "summary_size_target",
        0
    )

    actual = sequential_data.get(
        "final_summary_tokens",
        0
    )

    if target > 0:

        summary_size_usage = (
            actual
            / target
            * 100
        )

        print(
            f"Summary size usage: "
            f"{summary_size_usage:.2f}%"
        )

else:

    print(
        "No sequential output found."
    )


print("\n" + "=" * 60)
print("KEY SENTENCES EVALUATION")
print("=" * 60)

print(
    "Extracted sentences:",
    total_extracted
)

print(
    "Validated sentences:",
    total_validated
)

print(
    "Rejected sentences:",
    total_rejected
)

print(
    f"Exact match rate: "
    f"{exact_match_rate:.2f}%"
)


print("\n" + "=" * 60)
print("CLAIMS EVALUATION")
print("=" * 60)

print(
    "Total claims:",
    total_claims
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
    f"Supported rate: "
    f"{supported_rate:.2f}%"
)

print("\n" + "=" * 80)
print("EFFICIENCY COMPARISON")
print("=" * 80)

sequential_runtime = 158.99
sequential_calls = 12

key_sentences_runtime = 282.51
key_sentences_calls = 3

claims_runtime = 214.50
claims_calls = 6

print("\nSequential")
print("Runtime:", sequential_runtime, "seconds")
print("Model calls:", sequential_calls)

print("\nKey Sentences")
print("Runtime:", key_sentences_runtime, "seconds")
print("Model calls:", key_sentences_calls)

print("\nClaims")
print("Runtime:", claims_runtime, "seconds")
print("Model calls:", claims_calls)