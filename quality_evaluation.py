import json
import os

from jinja2 import Environment, FileSystemLoader

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


# ============================================================
# SETTINGS
# ============================================================

DOCUMENT_PATH = "data/ai_act.pdf"
CONTEXT_SIZE = 4000
MAX_CHUNKS = 3


# ============================================================
# LOAD SOURCE
# ============================================================

document_text = load_document(
    DOCUMENT_PATH
)

chunks = chunk_document(
    document_text,
    max_tokens=CONTEXT_SIZE
)

test_chunks = chunks[:MAX_CHUNKS]

source_text = "\n\n".join(
    test_chunks
)


# ============================================================
# LOAD EVALUATION PROMPT
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

evaluation_template = env.get_template(
    "evaluate_quality.j2"
)


# ============================================================
# HELPER: RUN QUALITY EVALUATION
# ============================================================

def evaluate_output(
    name,
    output_text
):

    print(
        f"\nEvaluating {name}..."
    )

    prompt = evaluation_template.render(
        source_text=source_text,
        output_text=output_text
    )

    result = generate_summary(
        prompt
    )

    try:
        parsed = json.loads(
            result
        )

    except json.JSONDecodeError:

        print(
            f"Warning: invalid JSON for {name}"
        )

        print(result)

        return None

    return parsed


# ============================================================
# LOAD SEQUENTIAL OUTPUT
# ============================================================

sequential_path = os.path.join(
    "outputs",
    "sequential",
    "result.json"
)

with open(
    sequential_path,
    "r",
    encoding="utf-8"
) as f:

    sequential_data = json.load(f)

sequential_output = sequential_data[
    "final_summary"
]


# ============================================================
# LOAD KEY SENTENCES OUTPUT
# ============================================================

key_sentences_dir = os.path.join(
    "outputs",
    "key_sentences",
    "validated"
)

key_sentence_parts = []

for i in range(
    1,
    MAX_CHUNKS + 1
):

    path = os.path.join(
        key_sentences_dir,
        f"chunk_{i}.txt"
    )

    if os.path.exists(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            key_sentence_parts.append(
                f.read().strip()
            )

key_sentences_output = "\n".join(
    key_sentence_parts
)


# ============================================================
# LOAD CLAIMS OUTPUT
# ============================================================

claims_dir = os.path.join(
    "outputs",
    "claims",
    "extracted"
)

claim_parts = []

for i in range(
    1,
    MAX_CHUNKS + 1
):

    path = os.path.join(
        claims_dir,
        f"chunk_{i}.json"
    )

    if not os.path.exists(path):
        continue

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    claims = data.get(
        "claims",
        []
    )

    claim_parts.extend(
        claims
    )

claims_output = "\n".join(
    claim_parts
)


# ============================================================
# RUN EVALUATIONS
# ============================================================

results = {}

results["sequential"] = evaluate_output(
    "Sequential",
    sequential_output
)

results["key_sentences"] = evaluate_output(
    "Key Sentences",
    key_sentences_output
)

results["claims"] = evaluate_output(
    "Claims",
    claims_output
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("QUALITY EVALUATION RESULTS")
print("=" * 70)


for approach, result in results.items():

    print(
        f"\n{approach.upper()}"
    )

    if result is None:
        print(
            "Evaluation failed."
        )
        continue

    faithfulness = result.get(
        "faithfulness",
        {}
    )

    coverage = result.get(
        "coverage",
        {}
    )

    coherence = result.get(
        "coherence",
        {}
    )

    print(
        "Faithfulness:",
        faithfulness.get("score"),
        "/ 5"
    )

    print(
        "Coverage:",
        coverage.get("score"),
        "/ 5"
    )

    print(
        "Coherence:",
        coherence.get("score"),
        "/ 5"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = os.path.join(
    "outputs",
    "evaluation"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "quality_results.json"
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
    "\nQuality evaluation saved:",
    output_path
)