import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Show non-supported statements from aggregated verification."
)

parser.add_argument(
    "verification_file",
    help="Path to verification JSON file"
)

args = parser.parse_args()


path = Path(args.verification_file)

data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


aggregation = data.get(
    "statement_aggregation",
    {}
)

found = False

for statement_id, item in aggregation.items():

    status = item.get(
        "final_status"
    )

    if status == "supported":
        continue

    found = True

    print()
    print("=" * 70)

    print(
        f"Statement {statement_id}"
    )

    print(
        "Status:",
        status
    )

    print()
    print(
        item.get(
            "statement",
            ""
        )
    )

    print()
    print(
        "Supporting chunks:",
        item.get(
            "supporting_chunks",
            []
        )
    )

    print(
        "Partial chunks:",
        item.get(
            "partial_chunks",
            []
        )
    )

    print(
        "Contradicted chunks:",
        item.get(
            "contradicted_chunks",
            []
        )
    )


if not found:

    print(
        "No problematic statements found."
    )