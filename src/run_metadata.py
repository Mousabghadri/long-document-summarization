
from __future__ import annotations

import json
import time
from pathlib import Path

from src.chunking import count_tokens
from src.llm import get_usage, llm_settings

METADATA_SCHEMA = 2


def write_metadata(output_dir, method: str, document_path: str, document_text: str,
                   start_time: float, target_words: int, final_words: int,
                   n_chunks: int, context_size: int, extra: dict = None) -> Path:
    usage = get_usage()
    data = {
        "metadata_schema": METADATA_SCHEMA,
        "method": method,
        "document": str(document_path),
        "document_tokens": count_tokens(document_text),
        "document_words": len(document_text.split()),
        "context_size": context_size,
        "chunks": n_chunks,
        "target_words": target_words,
        "final_words": final_words,
        "runtime_seconds": round(time.time() - start_time, 2),
        "llm_settings": llm_settings(),
        "usage": usage,
        # kept for backward compatibility with older readers
        "model_calls": usage["api_requests"],
    }
    if extra:
        data.update(extra)
    path = Path(output_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_run_summary(method: str, final_words: int, start_time: float):
    usage = get_usage()
    print("\n" + "=" * 80)
    print(f"{method.upper()} COMPLETE")
    print("=" * 80)
    print("Final summary words:", final_words)
    print(f"Runtime: {time.time() - start_time:.2f} seconds")
    print("Model calls:", usage["api_requests"])
    print("Failed requests:", usage["failed_requests"])
    print("Prompt tokens:", usage["prompt_tokens"])
    print("Completion tokens:", usage["completion_tokens"])
