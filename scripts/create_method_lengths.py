import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create different length outputs while preserving exact units"
)

parser.add_argument(
    "file",
    help="Input TXT file containing one sentence/claim per line"
)

parser.add_argument(
    "--output-prefix",
    required=True,
    help="Example: ai_act_key_sentences"
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


def select_evenly(units, target_words):
    """
    Select complete units from across the whole document
    while staying approximately within the target word count.
    """

    total_words = sum(count_words(unit) for unit in units)

    # If the whole output is already shorter than target,
    # keep everything.
    if total_words <= target_words:
        return units[:]

    selected_indexes = []
    selected_words = 0

    # Try units in evenly distributed order across the document
    total_units = len(units)

    for step in range(total_units):

        index = round(
            step * (total_units - 1) / max(total_units - 1, 1)
        )

        if index in selected_indexes:
            continue

        unit_words = count_words(
            units[index]
        )

        if selected_words + unit_words <= target_words:
            selected_indexes.append(
                index
            )

            selected_words += unit_words

    # Restore original document order
    selected_indexes.sort()

    return [
        units[index]
        for index in selected_indexes
    ]


# ============================================================
# LOAD INPUT
# ============================================================

with open(
    args.file,
    "r",
    encoding="utf-8"
) as f:

    units = [
        line.strip()
        for line in f
        if line.strip()
    ]


print("Total units:", len(units))

full_words = sum(
    count_words(unit)
    for unit in units
)

print("Full output words:", full_words)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = os.path.dirname(
    args.file
)

if not output_dir:
    output_dir = "."


# ============================================================
# CREATE LENGTHS
# ============================================================

targets = [
    2000,
    1000,
    500,
    250
]


for target in targets:

    selected = select_evenly(
        units,
        target
    )

    output_text = "\n".join(
        selected
    )

    final_words = count_words(
        output_text
    )

    output_path = os.path.join(
        output_dir,
        f"{args.output_prefix}_{target}_words.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            output_text
        )

    print(
        f"Target {target}: "
        f"{final_words} words | "
        f"{len(selected)} units"
    )

    print(
        "Saved:",
        output_path
    )