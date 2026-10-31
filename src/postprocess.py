from __future__ import annotations

import os

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

LENGTH_TOLERANCE = 0.10


def prompt_env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
    # SUMMARY_LANGUAGE (e.g. "German"): write every summary in that language
    env.globals["output_language"] = os.getenv("SUMMARY_LANGUAGE", "").strip()
    return env


def count_words(text: str) -> int:
    return len(text.split())


def clean_plain_text(text: str) -> str:
    """Remove Markdown artefacts so every method is scored on plain text."""
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        if re.fullmatch(r"[-=*_]{3,}", stripped):
            continue

        stripped = re.sub(r"^#{1,6}\s*", "", stripped)

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            stripped = f"{cells[0]}: {cells[1]}" if len(cells) == 2 else " - ".join(cells)

        stripped = re.sub(r"^[-*+•]\s+", "- ", stripped)
        stripped = stripped.replace("**", "").replace("__", "")
        stripped = re.sub(r"(?<!\*)\*(?!\*)", "", stripped)
        stripped = re.sub(r"(?<!_)_(?!_)", "", stripped)
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")
_ABBREV = re.compile(r"\b(Art|Arts|No|Nos|e\.g|i\.e|cf|para|paras|p|pp|v|vs|Inc|Co|Ltd|U\.S|Mr|Mrs|Dr|St|Reg|Dir)\.$")


def split_sentences(paragraph: str) -> list:
    """Sentence splitter that does not break on common legal abbreviations."""
    raw = _SENT_SPLIT.split(paragraph.strip())
    out = []
    for piece in raw:
        if out and _ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + piece
        else:
            out.append(piece)
    return [s.strip() for s in out if s.strip()]


def truncate_to_word_limit(text: str, max_words: int) -> str:
    """Keep whole sentences (and paragraph breaks) up to max_words."""
    if count_words(text) <= max_words:
        return text.strip()

    kept_paragraphs = []
    total = 0
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        kept = []
        for sentence in split_sentences(paragraph):
            n = len(sentence.split())
            if total + n > max_words:
                break
            kept.append(sentence)
            total += n
        else:
            if kept:
                kept_paragraphs.append(" ".join(kept))
            continue
        if kept:
            kept_paragraphs.append(" ".join(kept))
        break

    result = "\n\n".join(kept_paragraphs).strip()
    if not result:  # first sentence alone is longer than the budget
        result = " ".join(text.split()[:max_words])
    # never end on a dangling list number such as "3."
    result = re.sub(r"\n?\s*\d+\.\s*$", "", result).strip()
    return result


def finalize_summary(text: str, target_words: int, compress_call, tolerance: float = LENGTH_TOLERANCE):
    """
    Apply the shared final length control.

    compress_call(prompt) -> str must be the method's own LLM call function, so
    the extra call is counted in that method's usage.

    Returns (final_text, info_dict).
    """
    info = {"initial_words": None, "compression_calls": 0, "truncated": False}
    text = clean_plain_text(text)
    info["initial_words"] = count_words(text)

    if count_words(text) > int(target_words * (1 + tolerance)):
        prompt = prompt_env().get_template("compress.j2").render(
            summary=text, summary_length=target_words
        )
        compressed = clean_plain_text(compress_call(prompt))
        info["compression_calls"] = 1
        if count_words(compressed) < count_words(text):
            text = compressed

    if count_words(text) > target_words:
        text = truncate_to_word_limit(text, target_words)
        info["truncated"] = True

    info["final_words"] = count_words(text)
    return text, info
