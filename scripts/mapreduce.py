
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
from src.hierarchical import add_overlap, summarize_chunks, merge_groups, merge_one, merge_tokens, save_stage
from src.run_metadata import write_metadata, print_run_summary

parser = argparse.ArgumentParser(description="Map-Reduce long-document summarization")
parser.add_argument("file", help="Path to the document file")
parser.add_argument("--context-size", type=int, default=8000, help="Maximum source tokens per chunk")
parser.add_argument("--summary-words", type=int, default=1000, help="Target final summary length")
parser.add_argument("--chunk-summary-words", type=int, default=500, help="Target length of each map summary")
parser.add_argument("--intermediate-summary-words", type=int, default=800, help="Target length of collapsed summaries")
parser.add_argument("--max-input-tokens", type=int, default=100000,
                    help="Largest prompt (tokens) allowed for one reduce call")
parser.add_argument("--workers", type=int, default=5, help="Maximum parallel requests")
parser.add_argument("--overlap-tokens", type=int, default=0,
                    help="tokens of the previous chunk repeated at the start of each chunk (0 = none)")
parser.add_argument("--output-name", required=True, help="Output name")
args = parser.parse_args()

document_text = load_document(args.file)
chunks = chunk_document(document_text, max_tokens=args.context_size)
if not chunks:
    raise RuntimeError("No chunks were created.")
chunks = add_overlap(chunks, args.overlap_tokens)
print("Chunks:", len(chunks), "| map words:", args.chunk_summary_words,
      "| final words:", args.summary_words, "| max reduce input:", args.max_input_tokens)

output_dir = Path("outputs") / "method_outputs" / args.output_name
intermediate_dir = output_dir / "intermediate"
intermediate_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()
collapse_rounds = 0


def pack_by_tokens(summaries, budget):
    groups, cur = [], []
    for s in summaries:
        if cur and merge_tokens(cur + [s]) > budget:
            groups.append(cur)
            cur = [s]
        else:
            cur.append(s)
    if cur:
        groups.append(cur)
    return groups


if len(chunks) == 1:
    print("Document fits in one chunk: summarizing it directly at the final length.")
    reduced = generate_summary(
        prompt_env().get_template("summarize.j2").render(chunk=chunks[0],
                                                         summary_length=args.summary_words),
        purpose="single_chunk_final")
else:
    print("\nMAP")
    current = summarize_chunks(chunks, args.chunk_summary_words, args.workers)
    save_stage(intermediate_dir, 1, current)

    stage = 2
    while merge_tokens(current) > args.max_input_tokens:
        groups = pack_by_tokens(current, args.max_input_tokens)
        if len(groups) == len(current):
            raise RuntimeError("Single summaries exceed --max-input-tokens; increase it.")
        print(f"\nCOLLAPSE {stage - 1}: {len(current)} summaries -> {len(groups)}")
        current = merge_groups(groups, args.intermediate_summary_words, args.workers,
                               purpose="mapreduce_collapse")
        save_stage(intermediate_dir, stage, current)
        collapse_rounds += 1
        stage += 1

    print(f"\nREDUCE: {len(current)} summaries -> final ({args.summary_words} words)")
    reduced = merge_one(current, args.summary_words, purpose="mapreduce_reduce")
    save_stage(intermediate_dir, stage, [reduced])

final_summary, final_info = finalize_summary(
    reduced, args.summary_words,
    compress_call=lambda p: generate_summary(p, purpose="final_compress"))
final_words = count_words(final_summary)

summary_path = output_dir / f"{args.output_name}_mapreduce_summary_{args.summary_words}_words.txt"
summary_path.write_text(final_summary, encoding="utf-8")

meta_path = write_metadata(
    output_dir, "mapreduce", args.file, document_text, start_time,
    args.summary_words, final_words, len(chunks), args.context_size,
    extra={
        "workers": args.workers,
        "chunk_summary_words": args.chunk_summary_words,
        "intermediate_summary_words": args.intermediate_summary_words,
        "max_input_tokens": args.max_input_tokens,
        "collapse_rounds": collapse_rounds,
        "overlap_tokens": args.overlap_tokens,
        "final_length_control": final_info,
    })

print_run_summary("map-reduce", final_words, start_time)
print("Summary saved:", summary_path)
print("Metadata saved:", meta_path)