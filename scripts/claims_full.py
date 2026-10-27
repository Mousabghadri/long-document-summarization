import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary, generate_json
from src.postprocess import count_words, finalize_summary, prompt_env
from src.run_metadata import write_metadata, print_run_summary


parser = argparse.ArgumentParser(description="Extract, verify, and summarize claims from a full document")
parser.add_argument("file", help="Path to the document file")
parser.add_argument("--context-size", type=int, default=8000, help="Maximum source tokens per chunk")
parser.add_argument("--max-claims", type=int, default=10, help="Maximum claims per chunk")
parser.add_argument("--summary-words", type=int, default=1000, help="Target final summary length in words")
parser.add_argument("--additional-instructions", type=str, default="",
                    help="Optional extra instructions (do NOT use in the fair comparison)")
parser.add_argument("--output-name", required=True, help="Output name, e.g. ai_act or gdpr")
parser.add_argument("--use-themes", action="store_true",
                    help="Experimental document-level theme guidance (NOT part of the fair comparison)")
parser.add_argument("--workers", type=int, default=5, help="Maximum number of chunks processed in parallel")
args = parser.parse_args()

document_text = load_document(args.file)
chunks = chunk_document(document_text, max_tokens=args.context_size)
if not chunks:
    raise ValueError("No chunks were created from the document.")

print("Total chunks:", len(chunks))
print("Context size:", args.context_size)
print("Target summary:", args.summary_words, "words")
print("Parallel workers:", args.workers)
print("Theme guidance:", "ENABLED" if args.use_themes else "DISABLED")

env = prompt_env()
extract_template = env.get_template("extract_claims.j2")
verify_template = env.get_template("verify_claims.j2")
summary_template = env.get_template("summarize_from_claims.j2")
themes_template = env.get_template("extract_document_themes.j2") if args.use_themes else None

output_dir = os.path.join("outputs", "method_outputs", args.output_name)
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
themes_text = ""
document_themes = []

if args.use_themes:
    if len(chunks) <= 2:
        sample_chunks = chunks
    else:
        sample_chunks = [chunks[0], chunks[len(chunks) // 2], chunks[-1]]
    print("\nExtracting document-level themes...")
    try:
        themes_json = generate_json(
            themes_template.render(document_sample="\n\n".join(sample_chunks), max_themes=8),
            purpose="claims_themes")
        document_themes = themes_json.get("themes", [])
    except RuntimeError as e:
        print("Theme extraction failed:", e)
    themes_text = "\n".join(f"- {t}" for t in document_themes)


def _norm(s):
    return " ".join(str(s).split()).lower()


def process_chunk(index, chunk):
    chunk_number = index + 1
    print(f"\nStarting chunk {chunk_number}/{len(chunks)}...")

    extracted = generate_json(
        extract_template.render(source_text=chunk, max_claims=args.max_claims,
                                document_themes=themes_text, use_themes=args.use_themes),
        purpose="claims_extract")
    claims = [str(c).strip() for c in extracted.get("claims", []) if str(c).strip()]
    claims = claims[:args.max_claims]  # budget enforced deterministically
    print(f"Chunk {chunk_number}: extracted {len(claims)} claims")

    counts = {"supported": 0, "partial": 0, "unsupported": 0, "contradicted": 0}
    if not claims:
        return {"index": index, "claims": [], "supported_claims": [], **counts}

    verified = generate_json(
        verify_template.render(source_text=chunk, summary="\n".join(claims)),
        purpose="claims_verify")

    items = [it for it in verified.get("claims", []) if isinstance(it, dict)]
    labels = {}
    for item in items:
        labels.setdefault(_norm(item.get("claim", "")), item.get("label"))
    positional = len(items) == len(claims)

    supported_claims = []
    for pos, claim in enumerate(claims):  # keep only claims that were actually extracted
        label = labels.get(_norm(claim))
        if label is None and positional:  # verifier slightly re-worded the claim text
            label = items[pos].get("label")
        label = label or "unsupported"
        if label not in counts:
            label = "unsupported"
        counts[label] += 1
        if label == "supported":
            supported_claims.append(claim)

    print(f"Chunk {chunk_number} complete - " +
          ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return {"index": index, "claims": claims, "supported_claims": supported_claims, **counts}


print("\nStarting parallel Claims extraction + verification...")
results = []
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}
    for future in as_completed(futures):
        i = futures[future]
        try:
            results.append(future.result())
        except Exception as e:
            raise RuntimeError(f"Chunk {i + 1} failed: {e}") from e

results.sort(key=lambda r: r["index"])
all_supported = [c for r in results for c in r["supported_claims"]]
totals = {k: sum(r[k] for r in results) for k in ("supported", "partial", "unsupported", "contradicted")}
total_extracted = sum(len(r["claims"]) for r in results)

claims_path = os.path.join(output_dir, f"{args.output_name}_supported_claims.txt")
with open(claims_path, "w", encoding="utf-8") as f:
    for claim in all_supported:
        f.write(claim + "\n")

if not all_supported:
    raise RuntimeError("No supported claims - cannot build a summary.")

print("\nGenerating final summary from supported claims...")
draft = generate_summary(
    summary_template.render(claims_text="\n".join(all_supported), document_themes=themes_text,
                            use_themes=args.use_themes, summary_length=args.summary_words,
                            additional_instructions=args.additional_instructions),
    purpose="final_summary")

final_summary, final_info = finalize_summary(
    draft, args.summary_words,
    compress_call=lambda p: generate_summary(p, purpose="final_compress"))
final_words = count_words(final_summary)

summary_path = os.path.join(
    output_dir, f"{args.output_name}_claims_summary_{args.summary_words}_words.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(final_summary)

meta_path = write_metadata(
    output_dir, "claims", args.file, document_text, start_time,
    args.summary_words, final_words, len(chunks), args.context_size,
    extra={
        "workers": args.workers,
        "max_claims_per_chunk": args.max_claims,
        "use_themes": args.use_themes,
        "document_themes": document_themes,
        "extracted_claims": total_extracted,
        "claim_labels": totals,
        "support_words": count_words("\n".join(all_supported)),
        "final_length_control": final_info,
        "additional_instructions": args.additional_instructions or None,
    })

print_run_summary("claims", final_words, start_time)
print("Claim labels:", totals)
print("Claims saved:", claims_path)
print("Summary saved:", summary_path)
print("Metadata saved:", meta_path)
