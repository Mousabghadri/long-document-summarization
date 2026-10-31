from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.chunking import count_tokens
from src.llm import generate_summary
from src.postprocess import (
    clean_plain_text,
    count_words,
    prompt_env,
    truncate_to_word_limit,
)


_env = prompt_env()

_summarize = _env.get_template("summarize.j2")
_merge = _env.get_template("summn_merge.j2")


def _enforce(text, words, label):

    text = clean_plain_text(text)

    if count_words(text) > words:
        print(
            f"{label}: {count_words(text)} words -> "
            f"truncating to <= {words}"
        )

        text = truncate_to_word_limit(text, words)

    return text


def add_overlap(chunks, overlap_tokens):
    """
    Prefix every chunk (except the first) with the last complete sentences of
    the previous chunk, up to `overlap_tokens`, so a provision split at a chunk
    border is seen whole by one of the two chunks.  0 = no overlap.
    """

    if overlap_tokens <= 0 or len(chunks) < 2:
        return list(chunks)

    import re

    out = [chunks[0]]

    for prev, cur in zip(chunks, chunks[1:]):

        sentences = re.split(r"(?<=[.;:!?])\s+", prev.strip())

        tail, used = [], 0

        for sent in reversed(sentences):

            n = count_tokens(sent)

            if used + n > overlap_tokens:
                break

            tail.insert(0, sent)
            used += n

        if not tail:  # one very long last sentence: keep its end
            words = prev.split()

            tail = [
                " ".join(words[-max(1, overlap_tokens // 2):])
            ]

        out.append(
            " ".join(tail) + "\n\n" + cur
        )

    return out


def summarize_chunks(chunks, words, workers, purpose="map_chunk"):
    """Stage 1: summarize every chunk independently (in parallel)."""

    def one(i, chunk):

        print(f"Stage 1: chunk {i + 1}/{len(chunks)}")

        out = generate_summary(
            _summarize.render(
                chunk=chunk,
                summary_length=words
            ),
            purpose=purpose
        )

        return i, _enforce(out, words, f"Chunk {i + 1}")

    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=workers) as pool:

        futures = {
            pool.submit(one, i, c): i
            for i, c in enumerate(chunks)
        }

        for fut in as_completed(futures):

            i = futures[fut]

            try:
                idx, text = fut.result()

            except Exception as e:
                raise RuntimeError(
                    f"Chunk {i + 1} failed: {e}"
                ) from e

            results[idx] = text

    return results


def merge_text(summaries):

    return "\n\n".join(
        f"SECTION SUMMARY {i + 1}:\n{s}"
        for i, s in enumerate(summaries)
    )


def merge_tokens(summaries) -> int:

    return count_tokens(
        _merge.render(
            summaries_text=merge_text(summaries),
            summary_length=1000
        )
    )


def merge_one(summaries, words, purpose="merge"):
    """Merge several consecutive summaries into one (single LLM call)."""

    out = generate_summary(
        _merge.render(
            summaries_text=merge_text(summaries),
            summary_length=words
        ),
        purpose=purpose
    )

    return _enforce(out, words, purpose)


def merge_groups(groups, words, workers, purpose):
    """Merge each group in parallel; a single-member group passes through."""

    def one(i, group):

        if len(group) == 1:
            return i, _enforce(
                group[0],
                words,
                f"{purpose} group {i + 1}"
            )

        return i, merge_one(
            group,
            words,
            purpose=purpose
        )

    results = [None] * len(groups)

    with ThreadPoolExecutor(max_workers=workers) as pool:

        futures = {
            pool.submit(one, i, g): i
            for i, g in enumerate(groups)
        }

        for fut in as_completed(futures):

            i = futures[fut]

            try:
                idx, text = fut.result()

            except Exception as e:
                raise RuntimeError(
                    f"{purpose} group {i + 1} failed: {e}"
                ) from e

            results[idx] = text

    return results


def save_stage(intermediate_dir: Path, stage: int, summaries):

    d = Path(intermediate_dir) / f"stage_{stage}"

    d.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, s in enumerate(summaries, 1):

        (d / f"summary_{i}.txt").write_text(
            s,
            encoding="utf-8"
        )