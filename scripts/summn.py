
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary
from src.postprocess import count_words, finalize_summary, prompt_env
from src.hierarchical import summarize_chunks, merge_groups, save_stage
from src.run_metadata import write_metadata, print_run_summary

parser = argparse.ArgumentParser(description="SummN-style multi-stage long-document summarization")
parser.add_argument("file", help="Path to the document file")
parser.add_argument("--context-size", type=int, default=8000, help="Maximum source tokens per initial chunk")
parser.add_argument("--summary-words", type=int, default=1000, help="Target final summary length")
parser.add_argument("--chunk-summary-words", type=int, default=500, help="Target length of each stage-1 chunk summary")
parser.add_argument("--intermediate-summary-words", type=int, default=800, help="Target length of intermediate merged summaries")
parser.add_argument("--merge-group-size", type=int, default=4, help="Number of summaries merged together at each stage")
parser.add_argument("--workers", type=int, default=5, help="Maximum parallel requests per stage")
parser.add_argument("--output-name", required=True, help="Output name")
args = parser.parse_args()

if args.merge_group_size < 2:
    sys.exit("--merge-group-size must be >= 2")

document_text = load_document(args.file)
chunks = chunk_document(document_text, max_tokens=args.context_size)
if not chunks:
    raise RuntimeError("No chunks were created.")

print("Initial chunks:", len(chunks))
print("Chunk summary target:", args.chunk_summary_words, "| intermediate:",
      args.intermediate_summary_words, "| final:", args.summary_words, "words")
print("Merge group size:", args.merge_group_size, "| workers:", args.workers)

output_dir = Path("outputs") / "method_outputs" / args.output_name
intermediate_dir = output_dir / "intermediate"
intermediate_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()
stage_sizes = []

if len(chunks) == 1:
    print("Document fits in one chunk: summarizing it directly at the final length.")
    current = [generate_summary(
        prompt_env().get_template("summarize.j2").render(chunk=chunks[0],
                                                         summary_length=args.summary_words),
        purpose="single_chunk_final")]
    merge_stages = 0
else:
    print("\nSTAGE 1: parallel chunk summarization")
    current = summarize_chunks(chunks, args.chunk_summary_words, args.workers)
    save_stage(intermediate_dir, 1, current)
    stage_sizes.append(len(current))

    stage = 2
    while len(current) > 1:
        g = args.merge_group_size
        groups = [current[i:i + g] for i in range(0, len(current), g)]
        final = len(groups) == 1
        words = args.summary_words if final else args.intermediate_summary_words
        print(f"\nSTAGE {stage}: {len(current)} summaries -> {len(groups)} "
              f"({'FINAL' if final else 'INTERMEDIATE'}, {words} words)")
        current = merge_groups(groups, words, args.workers, purpose=f"summn_stage_{stage}")
        save_stage(intermediate_dir, stage, current)
        stage_sizes.append(len(current))
        stage += 1
    merge_stages = stage - 2

final_summary, final_info = finalize_summary(
    current[0], args.summary_words,
    compress_call=lambda p: generate_summary(p, purpose="final_compress"))
final_words = count_words(final_summary)

summary_path = output_dir / f"{args.output_name}_summn_summary_{args.summary_words}_words.txt"
summary_path.write_text(final_summary, encoding="utf-8")

meta_path = write_metadata(
    output_dir, "summn", args.file, document_text, start_time,
    args.summary_words, final_words, len(chunks), args.context_size,
    extra={
        "workers": args.workers,
        "chunk_summary_words": args.chunk_summary_words,
        "intermediate_summary_words": args.intermediate_summary_words,
        "merge_group_size": args.merge_group_size,
        "merge_stages": merge_stages,
        "summaries_per_stage": stage_sizes,
        "final_length_control": final_info,
    })

print_run_summary("summn-style", final_words, start_time)
print("Merge stages:", merge_stages)
print("Summary saved:", summary_path)
print("Metadata saved:", meta_path)
