
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary, get_usage
from src.postprocess import (
    clean_plain_text, count_words, finalize_summary, prompt_env, truncate_to_word_limit,
    LENGTH_TOLERANCE,
)
from src.run_metadata import write_metadata, print_run_summary


parser = argparse.ArgumentParser(description="Sequential (refine) long-document summarization")
parser.add_argument("file", help="Path to the document file")
parser.add_argument("--summary-words", type=int, default=1000, help="Target summary length in words")
parser.add_argument("--context-size", type=int, default=8000, help="Maximum source tokens per chunk")
parser.add_argument("--output-name", required=True, help="Output name, e.g. ai_act")
parser.add_argument("--no-resume", action="store_true",
                    help="Ignore saved intermediate progress and start from chunk 1")
args = parser.parse_args()

SUMMARY_WORDS = args.summary_words


def call_model(prompt, purpose):
    return generate_summary(prompt, purpose=purpose).strip()


document_text = load_document(args.file)
chunks = chunk_document(document_text, max_tokens=args.context_size)
if not chunks:
    raise ValueError("No chunks were created from the document.")

print("Total chunks:", len(chunks))
print("Context size:", args.context_size, "tokens")
print("Target summary:", SUMMARY_WORDS, "words")

env = prompt_env()
summarize_template = env.get_template("summarize.j2")
update_template = env.get_template("update.j2")
compress_template = env.get_template("compress.j2")

base_output_dir = os.path.join("outputs", "full_summaries", args.output_name)
intermediate_dir = os.path.join(base_output_dir, f"intermediate_{SUMMARY_WORDS}_words")
os.makedirs(intermediate_dir, exist_ok=True)

step_compressions = 0


def compress_running_summary(summary, label):
    """Method-intrinsic per-step length control of the running summary."""
    global step_compressions
    summary = clean_plain_text(summary)
    for attempt in range(1, 3):
        current = count_words(summary)
        if current <= SUMMARY_WORDS:
            return summary
        if current <= int(SUMMARY_WORDS * (1 + LENGTH_TOLERANCE)):
            print(f"{label}: small overshoot ({current} words) -> sentence-safe truncation")
            return truncate_to_word_limit(summary, SUMMARY_WORDS)
        print(f"{label}: {current} words -> compression attempt {attempt}")
        compressed = clean_plain_text(call_model(
            compress_template.render(summary=summary, summary_length=SUMMARY_WORDS),
            purpose="sequential_step_compress"))
        step_compressions += 1
        if count_words(compressed) >= current:
            print("Compression did not shorten the summary; stopping.")
            break
        summary = compressed
    if count_words(summary) > SUMMARY_WORDS:
        summary = truncate_to_word_limit(summary, SUMMARY_WORDS)
    return summary


def find_latest_intermediate():
    latest_chunk, latest_path = 0, None
    for i in range(1, len(chunks) + 1):
        path = os.path.join(intermediate_dir, f"chunk_{i}.txt")
        if os.path.exists(path):
            latest_chunk, latest_path = i, path
        else:
            break
    return latest_chunk, latest_path


start_time = time.time()

latest_chunk, latest_path = (0, None) if args.no_resume else find_latest_intermediate()
resumed_from_chunk = None

if latest_chunk > 0:
    print(f"\nFound saved progress at chunk {latest_chunk}. Resuming. "
          "NOTE: runtime and call counts of this run are NOT comparable.")
    with open(latest_path, "r", encoding="utf-8") as f:
        current_summary = f.read()
    start_chunk_index = latest_chunk
    resumed_from_chunk = latest_chunk
else:
    print(f"\nProcessing chunk 1/{len(chunks)}...")
    current_summary = call_model(
        summarize_template.render(chunk=chunks[0], summary_length=SUMMARY_WORDS),
        purpose="sequential_first_chunk")
    current_summary = compress_running_summary(current_summary, "Chunk 1")
    with open(os.path.join(intermediate_dir, "chunk_1.txt"), "w", encoding="utf-8") as f:
        f.write(current_summary)
    print("Current summary words:", count_words(current_summary))
    start_chunk_index = 1

for chunk_index in range(start_chunk_index, len(chunks)):
    chunk_number = chunk_index + 1
    print(f"\nProcessing chunk {chunk_number}/{len(chunks)}...")
    updated = call_model(
        update_template.render(current_summary=current_summary, chunk=chunks[chunk_index],
                               summary_length=SUMMARY_WORDS),
        purpose="sequential_update")
    current_summary = compress_running_summary(updated, f"Chunk {chunk_number}")
    with open(os.path.join(intermediate_dir, f"chunk_{chunk_number}.txt"), "w",
              encoding="utf-8") as f:
        f.write(current_summary)
    print("Current summary words:", count_words(current_summary))


# Shared final length control (identical for every method)
final_summary, final_info = finalize_summary(
    current_summary, SUMMARY_WORDS,
    compress_call=lambda p: call_model(p, purpose="final_compress"))
final_word_count = count_words(final_summary)

summary_path = os.path.join(base_output_dir, f"{args.output_name}_{SUMMARY_WORDS}_words.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(final_summary)

usage_now = get_usage()
meta_path = write_metadata(
    base_output_dir, "sequential", args.file, document_text, start_time,
    SUMMARY_WORDS, final_word_count, len(chunks), args.context_size,
    extra={
        "resumed_from_chunk": resumed_from_chunk,
        "step_compression_calls": step_compressions,
        "final_length_control": final_info,
        # backward-compatible keys
        "model_calls_this_run": usage_now["api_requests"],
        "runtime_seconds_this_run": round(time.time() - start_time, 2),
    })

print_run_summary("sequential", final_word_count, start_time)
print("Final summary saved:", summary_path)
print("Metadata saved:", meta_path)
