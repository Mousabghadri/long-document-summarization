
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rouge_score import rouge_scorer  # noqa: E402

from src.postprocess import clean_plain_text, count_words, split_sentences  # noqa: E402


def load_text(path):
    return clean_plain_text(Path(path).read_text(encoding="utf-8").strip())


def sentences_of(text):
    out = []
    for para in text.split("\n"):
        out.extend(split_sentences(para))
    return [s for s in out if s.strip()]


def semantic_similarity(reference, summary, model_name):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def doc_vector(text):
        sents = sentences_of(text) or [text]
        emb = model.encode(sents, normalize_embeddings=True, convert_to_numpy=True)
        # weight sentences by length so many tiny fragments do not dominate
        weights = np.array([max(len(s.split()), 1) for s in sents], dtype=float)
        vec = (emb * weights[:, None]).sum(axis=0) / weights.sum()
        return vec / (np.linalg.norm(vec) + 1e-12)

    return float(doc_vector(reference) @ doc_vector(summary))


def bertscore(reference, summary, model_type):
    from bert_score import score

    p, r, f = score([summary], [reference], model_type=model_type, lang="en", verbose=False)
    return float(p[0]), float(r[0]), float(f[0])


def main():
    parser = argparse.ArgumentParser(description="Compare one generated summary against a human reference.")
    parser.add_argument("reference", help="Path to the human reference summary")
    parser.add_argument("summary", help="Path to the generated summary")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic similarity")
    parser.add_argument("--language", default="english",
                        help="non-English: Unicode word tokens, no English stemmer (keeps umlauts etc.)")
    parser.add_argument("--bertscore", action="store_true", help="Also compute BERTScore")
    parser.add_argument("--bertscore-model", default="allenai/longformer-large-4096")
    args = parser.parse_args()

    reference = load_text(args.reference)
    summary = load_text(args.summary)
    ref_words, sum_words = count_words(reference), count_words(summary)

    if args.language.lower() in ("english", "en"):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    else:
        # the default rouge-score tokenizer keeps only [a-z0-9]: it would split
        # "Verordnung über" into "verordnung", "ber" and drop every umlaut
        class UnicodeTokenizer:
            def tokenize(self, text):
                return re.findall(r"\w+", text.lower())
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"],
                                          tokenizer=UnicodeTokenizer())
    # rougeLsum expects one sentence per line - apply the SAME splitting to both texts
    scores = scorer.score("\n".join(sentences_of(reference)), "\n".join(sentences_of(summary)))

    result = {
        "method": args.method,
        "human_reference_path": args.reference,
        "summary_path": args.summary,
        "human_reference_words": ref_words,
        "summary_words": sum_words,
        "length_ratio": round(sum_words / ref_words, 3) if ref_words else 0,
        "language": args.language,
    }
    for key, name in [("rouge1", "rouge_1"), ("rouge2", "rouge_2"), ("rougeL", "rouge_l"),
                      ("rougeLsum", "rouge_lsum")]:
        s = scores[key]
        result[f"{name}_f1"] = round(s.fmeasure, 4)
        result[f"{name}_precision"] = round(s.precision, 4)
        result[f"{name}_recall"] = round(s.recall, 4)

    if not args.no_semantic:
        result["semantic_similarity"] = round(
            semantic_similarity(reference, summary, args.semantic_model), 4)
        result["semantic_method"] = f"length-weighted mean sentence embedding ({args.semantic_model})"

    if args.bertscore:
        p, r, f = bertscore(reference, summary, args.bertscore_model)
        result.update({"bertscore_precision": round(p, 4), "bertscore_recall": round(r, 4),
                       "bertscore_f1": round(f, 4), "bertscore_model": args.bertscore_model})

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
