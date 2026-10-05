import json
import os


# ============================================================
# RESULTS
# ============================================================

results = {
    "Sequential": {
        "faithfulness": 5,
        "coverage": 3,
        "coherence": 5,
        "runtime_seconds": 158.99,
        "model_calls": 12
    },

    "Key Sentences": {
        "faithfulness": 5,
        "coverage": 2,
        "coherence": 3,
        "runtime_seconds": 282.51,
        "model_calls": 3
    },

    "Claims": {
        "faithfulness": 5,
        "coverage": 3,
        "coherence": 5,
        "runtime_seconds": 214.50,
        "model_calls": 6
    }
}


# ============================================================
# PRINT TABLE
# ============================================================

print("\n" + "=" * 90)
print("FINAL COMPARISON")
print("=" * 90)

print(
    f"{'Approach':<20}"
    f"{'Faith.':<10}"
    f"{'Coverage':<10}"
    f"{'Coherence':<12}"
    f"{'Runtime':<12}"
    f"{'Calls':<8}"
)

print("-" * 90)

for approach, data in results.items():

    print(
        f"{approach:<20}"
        f"{data['faithfulness']:<10}"
        f"{data['coverage']:<10}"
        f"{data['coherence']:<12}"
        f"{data['runtime_seconds']:<12.2f}"
        f"{data['model_calls']:<8}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = "outputs/evaluation"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "comparison_results.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    "\nComparison saved:",
    output_path
)