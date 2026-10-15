import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


def load_text(path):
    return Path(path).read_text(encoding="utf-8").strip()


def count_words(text):
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(
        description="Compare one generated summary against a human reference."
    )

    parser.add_argument(
        "reference",
        help="Path to the human reference summary"
    )

    parser.add_argument(
        "summary",
        help="Path to the generated summary"
    )

    parser.add_argument(
        "--method",
        default="unknown",
        help="Method name, e.g. claims, key_sentences, sequential"
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Where to save the evaluation JSON"
    )

    args = parser.parse_args()

    # ============================================================
    # LOAD TEXTS
    # ============================================================

    human_reference = load_text(args.reference)
    summary_text = load_text(args.summary)

    human_words = count_words(human_reference)
    summary_words = count_words(summary_text)

    print("Human reference words:", human_words)
    print("Generated summary words:", summary_words)

    # ============================================================
    # ROUGE
    # ============================================================

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge_scores = rouge.score(
        human_reference,
        summary_text
    )

    rouge1 = rouge_scores["rouge1"].fmeasure
    rouge2 = rouge_scores["rouge2"].fmeasure
    rougeL = rouge_scores["rougeL"].fmeasure

    # ============================================================
    # SEMANTIC SIMILARITY
    # ============================================================

    print("Loading semantic similarity model...")

    semantic_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    human_embedding = semantic_model.encode(
        human_reference,
        convert_to_tensor=True
    )

    summary_embedding = semantic_model.encode(
        summary_text,
        convert_to_tensor=True
    )

    semantic_similarity = util.cos_sim(
        human_embedding,
        summary_embedding
    ).item()

    # ============================================================
    # LENGTH RATIO
    # ============================================================

    length_ratio = (
        summary_words / human_words
        if human_words > 0
        else 0
    )

    # ============================================================
    # RESULT
    # ============================================================

    result = {
        "method": args.method,
        "human_reference_path": args.reference,
        "summary_path": args.summary,
        "human_reference_words": human_words,
        "summary_words": summary_words,
        "length_ratio": round(length_ratio, 3),
        "rouge_1_f1": round(rouge1, 4),
        "rouge_2_f1": round(rouge2, 4),
        "rouge_l_f1": round(rougeL, 4),
        "semantic_similarity": round(
            semantic_similarity,
            4
        )
    }

    # ============================================================
    # PRINT
    # ============================================================

    print()
    print("=" * 70)
    print("STANDARD METRICS")
    print("=" * 70)

    print("Method:", args.method)
    print("Summary words:", summary_words)
    print("Length ratio:", result["length_ratio"])
    print("ROUGE-1:", result["rouge_1_f1"])
    print("ROUGE-2:", result["rouge_2_f1"])
    print("ROUGE-L:", result["rouge_l_f1"])
    print(
        "Semantic similarity:",
        result["semantic_similarity"]
    )

    # ============================================================
    # SAVE
    # ============================================================

    output_path = Path(args.output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()