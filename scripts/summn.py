import argparse
import hashlib
import os
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from jinja2 import Environment, FileSystemLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document
from src.chunking import chunk_document
from src.llm import generate_summary


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="SummN-style multi-stage long-document summarization"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=8000,
    help="Maximum source tokens per initial chunk"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=1000,
    help="Target final summary length"
)

parser.add_argument(
    "--chunk-summary-words",
    type=int,
    default=500,
    help="Target length for each first-stage chunk summary"
)

parser.add_argument(
    "--intermediate-summary-words",
    type=int,
    default=800,
    help="Target length for intermediate merged summaries"
)

parser.add_argument(
    "--merge-group-size",
    type=int,
    default=4,
    help="Number of summaries merged together at each stage"
)

parser.add_argument(
    "--workers",
    type=int,
    default=5,
    help="Maximum parallel requests per stage"
)

parser.add_argument(
    "--output-name",
    required=True,
    help="Output name, e.g. ai_act_summn"
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


def clean_plain_text(text):

    lines = text.splitlines()
    cleaned = []

    for line in lines:

        line = line.strip()

        line = re.sub(
            r"^#{1,6}\s*",
            "",
            line
        )

        line = line.replace(
            "**",
            ""
        )

        line = line.replace(
            "__",
            ""
        )

        cleaned.append(
            line
        )

    text = "\n".join(
        cleaned
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def truncate_to_word_limit(
    text,
    max_words
):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    selected = []
    total_words = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        words = sentence.split()

        if total_words + len(words) > max_words:
            break

        selected.append(
            sentence
        )

        total_words += len(
            words
        )

    return " ".join(
        selected
    ).strip()


def words_to_max_tokens(target_words):
    """
    Give the reasoning model enough output budget for
    both reasoning tokens and the visible summary.
    """

    return max(
        500,
        int(target_words * 3.0)
    )


model_calls = 0
model_calls_lock = Lock()


def call_model(prompt, max_tokens, max_retries=3):
    global model_calls

    current_limit = max_tokens

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[API attempt {attempt}/{max_retries}] "
                f"max_tokens={current_limit}"
            )

            result = generate_summary(
                prompt,
                max_tokens=current_limit,
                reasoning_effort="low"
            )

            if result is not None and result.strip():
                with model_calls_lock:
                    model_calls += 1

                return clean_plain_text(result)

        except RuntimeError as e:
            error_message = str(e).lower()

            print(
                f"Model error "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

            is_length_error = (
                "token limit" in error_message
                or "finish_reason=length" in error_message
                or "reached the token limit" in error_message
            )

            if is_length_error and attempt < max_retries:
                current_limit = int(current_limit * 1.5)
                print(
                    f"Retrying with max_tokens={current_limit}"
                )

        if attempt < max_retries:
            time.sleep(2)

    raise RuntimeError(
        "Model failed after multiple retries."
    )


def enforce_word_limit(
    text,
    target_words,
    label
):

    current_words = count_words(
        text
    )

    if current_words <= target_words:
        return text

    print(
        f"{label}: {current_words} words -> "
        f"truncating to <= {target_words}"
    )

    shortened = truncate_to_word_limit(
        text,
        target_words
    )

    print(
        f"{label}: after truncation = "
        f"{count_words(shortened)} words"
    )

    return shortened


# ============================================================
# LOAD DOCUMENT
# ============================================================

document_text = load_document(
    args.file
)

chunks = chunk_document(
    document_text,
    max_tokens=args.context_size
)

if not chunks:

    raise RuntimeError(
        "No chunks were created."
    )

print(
    "Initial chunks:",
    len(chunks)
)

print(
    "Context size:",
    args.context_size
)

print(
    "Chunk summary target:",
    args.chunk_summary_words,
    "words"
)

print(
    "Intermediate summary target:",
    args.intermediate_summary_words,
    "words"
)

print(
    "Final summary target:",
    args.summary_words,
    "words"
)

print(
    "Merge group size:",
    args.merge_group_size
)

print(
    "Parallel workers:",
    args.workers
)


# ============================================================
# LOAD PROMPTS
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

summarize_template = env.get_template(
    "summarize.j2"
)

merge_template = env.get_template(
    "summn_merge.j2"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = (
    Path("outputs")
    / "method_outputs"
    / args.output_name
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

intermediate_dir = (
    output_dir
    / "intermediate"
)

intermediate_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TIMER
# ============================================================

start_time = time.time()


# ============================================================
# STAGE 1: PARALLEL CHUNK SUMMARIZATION
# ============================================================

def summarize_chunk(
    index,
    chunk
):

    chunk_number = index + 1

    print(
        f"Starting chunk summary "
        f"{chunk_number}/{len(chunks)}..."
    )

    prompt = summarize_template.render(
        chunk=chunk,
        summary_length=args.chunk_summary_words
    )

    checkpoint_dir = intermediate_dir / "checkpoints_stage1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # The fingerprint includes the exact rendered prompt,
    # relevant run settings and local model configuration.
    env_file = PROJECT_ROOT / ".env"
    env_bytes = env_file.read_bytes() if env_file.exists() else b""

    llm_file = PROJECT_ROOT / "src" / "llm.py"
    llm_bytes = llm_file.read_bytes() if llm_file.exists() else b""

    fingerprint_data = json.dumps(
        {
            "prompt": prompt,
            "chunk_words": args.chunk_summary_words,
            "context_size": args.context_size,
            "model_env_hash": hashlib.sha256(env_bytes).hexdigest(),
            "llm_code_hash": hashlib.sha256(llm_bytes).hexdigest(),
            "model_env_override": os.environ.get("MODEL_NAME", ""),
            "reasoning_effort": "low",
            "token_budget_multiplier": 3.0,
            "retry_policy": "length_x1.5_v1",
        },
        sort_keys=True
    )

    fingerprint = hashlib.sha256(
        fingerprint_data.encode("utf-8")
    ).hexdigest()

    checkpoint_path = (
        checkpoint_dir
        / f"chunk_{chunk_number:03d}_{fingerprint[:16]}.json"
    )

    if checkpoint_path.exists():
        saved = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        if (
            saved.get("fingerprint") == fingerprint
            and isinstance(saved.get("summary"), str)
            and saved["summary"].strip()
        ):
            print(f"Chunk {chunk_number}: loaded from checkpoint")
            return {
                "index": index,
                "summary": saved["summary"]
            }

    summary = call_model(
        prompt,
        max_tokens=words_to_max_tokens(
            args.chunk_summary_words
        )
    )

    summary = enforce_word_limit(
        summary,
        args.chunk_summary_words,
        f"Chunk {chunk_number}"
    )

    checkpoint_data = {
        "fingerprint": fingerprint,
        "chunk_number": chunk_number,
        "summary": summary
    }

    temp_path = checkpoint_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(checkpoint_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    os.replace(temp_path, checkpoint_path)

    print(
        f"Chunk {chunk_number} complete - "
        f"{count_words(summary)} words"
    )

    return {
        "index": index,
        "summary": summary
    }


print(
    "\nSTAGE 1: Parallel chunk summarization"
)

stage_results = []

with ThreadPoolExecutor(
    max_workers=args.workers
) as executor:

    futures = {
        executor.submit(
            summarize_chunk,
            i,
            chunk
        ): i
        for i, chunk in enumerate(chunks)
    }

    for future in as_completed(
        futures
    ):

        index = futures[
            future
        ]

        try:

            result = future.result()

            stage_results.append(
                result
            )

        except Exception as e:

            raise RuntimeError(
                f"Chunk {index + 1} failed: {e}"
            ) from e


stage_results.sort(
    key=lambda x: x["index"]
)

current_summaries = [
    item["summary"]
    for item in stage_results
]


# ============================================================
# SAVE STAGE 1
# ============================================================

stage_1_dir = (
    intermediate_dir
    / "stage_1"
)

stage_1_dir.mkdir(
    exist_ok=True
)

for i, summary in enumerate(
    current_summaries,
    start=1
):

    (
        stage_1_dir
        / f"summary_{i}.txt"
    ).write_text(
        summary,
        encoding="utf-8"
    )


# ============================================================
# MULTI-STAGE MERGING
# ============================================================

stage_number = 2


def merge_group(
    group_index,
    group,
    total_groups,
    target_words,
    current_stage
):

    # A group with one summary does not need another LLM call.
    if len(group) == 1:

        summary = enforce_word_limit(
            group[0],
            target_words,
            f"Stage {current_stage} group {group_index + 1}"
        )

        print(
            f"Stage {current_stage}: "
            f"group {group_index + 1}/{total_groups} "
            f"contains one summary - passing through "
            f"({count_words(summary)} words)"
        )

        return {
            "index": group_index,
            "summary": summary
        }

    print(
        f"Stage {current_stage}: "
        f"starting group "
        f"{group_index + 1}/{total_groups}..."
    )

    summaries_text = "\n\n".join(
        f"SECTION SUMMARY {i + 1}:\n{summary}"
        for i, summary in enumerate(group)
    )

    prompt = merge_template.render(
        summaries_text=summaries_text,
        summary_length=target_words
    )

    merged = call_model(
        prompt,
        max_tokens=words_to_max_tokens(
            target_words
        )
    )

    merged = enforce_word_limit(
        merged,
        target_words,
        f"Stage {current_stage} group {group_index + 1}"
    )

    print(
        f"Stage {current_stage}: "
        f"group {group_index + 1} complete - "
        f"{count_words(merged)} words"
    )

    return {
        "index": group_index,
        "summary": merged
    }


while len(
    current_summaries
) > 1:

    groups = [
        current_summaries[
            i:i + args.merge_group_size
        ]
        for i in range(
            0,
            len(current_summaries),
            args.merge_group_size
        )
    ]

    is_final_stage = (
        len(groups) == 1
    )

    if is_final_stage:

        stage_target_words = (
            args.summary_words
        )

        stage_label = "FINAL"

    else:

        stage_target_words = (
            args.intermediate_summary_words
        )

        stage_label = "INTERMEDIATE"

    print()

    print(
        f"STAGE {stage_number}: "
        f"Merging {len(current_summaries)} summaries "
        f"into {len(groups)} groups "
        f"({stage_label}, target "
        f"{stage_target_words} words)"
    )

    merge_results = []

    current_stage = stage_number

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(
                merge_group,
                i,
                group,
                len(groups),
                stage_target_words,
                current_stage
            ): i
            for i, group in enumerate(
                groups
            )
        }

        for future in as_completed(
            futures
        ):

            index = futures[
                future
            ]

            try:

                result = future.result()

                merge_results.append(
                    result
                )

            except Exception as e:

                raise RuntimeError(
                    f"Stage {current_stage}, "
                    f"group {index + 1} failed: {e}"
                ) from e

    merge_results.sort(
        key=lambda x: x["index"]
    )

    current_summaries = [
        item["summary"]
        for item in merge_results
    ]

    stage_dir = (
        intermediate_dir
        / f"stage_{stage_number}"
    )

    stage_dir.mkdir(
        exist_ok=True
    )

    for i, summary in enumerate(
        current_summaries,
        start=1
    ):

        (
            stage_dir
            / f"summary_{i}.txt"
        ).write_text(
            summary,
            encoding="utf-8"
        )

    stage_number += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

final_summary = current_summaries[
    0
]

initial_final_words = count_words(
    final_summary
)

print(
    "\nInitial final summary words:",
    initial_final_words
)

final_summary = enforce_word_limit(
    final_summary,
    args.summary_words,
    "Final summary"
)

final_words = count_words(
    final_summary
)


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_path = (
    output_dir
    / f"{args.output_name}_summn_summary_{args.summary_words}_words.txt"
)

summary_path.write_text(
    final_summary,
    encoding="utf-8"
)


# ============================================================
# METADATA
# ============================================================

runtime = time.time() - start_time

metadata = {
    "method": "SummN-style",
    "document": args.file,
    "context_size": args.context_size,
    "initial_chunks": len(chunks),
    "chunk_summary_words": args.chunk_summary_words,
    "intermediate_summary_words": args.intermediate_summary_words,
    "merge_group_size": args.merge_group_size,
    "workers": args.workers,
    "final_target_words": args.summary_words,
    "final_words": final_words,
    "model_calls": model_calls,
    "runtime_seconds": runtime,
    "merge_stages": stage_number - 2
}

metadata_path = (
    output_dir
    / "metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# RESULTS
# ============================================================

print()
print(
    "=" * 80
)

print(
    "SUMMN-STYLE SUMMARIZATION COMPLETE"
)

print(
    "=" * 80
)

print(
    "Initial chunks:",
    len(chunks)
)

print(
    "Merge stages:",
    stage_number - 2
)

print(
    "Final summary words:",
    final_words
)

print(
    f"Runtime: {runtime:.2f} seconds"
)

print(
    "Model calls:",
    model_calls
)

print(
    "Parallel workers:",
    args.workers
)

print(
    "Summary saved:",
    summary_path
)

print(
    "Metadata saved:",
    metadata_path
)