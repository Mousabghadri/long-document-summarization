"""
Memorization probe: summarize an EU act from its TITLE ONLY.  If these
summaries score close to Direct against the reference, the model already knows
the act (or its official summary) from pre-training, and reference-based scores
partly measure memory rather than summarization.
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document  # noqa: E402
from src.eu_structure import title_of  # noqa: E402
from src.llm import generate_summary  # noqa: E402
from src.postprocess import count_words, finalize_summary, prompt_env  # noqa: E402
from src.run_metadata import write_metadata, print_run_summary  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("--summary-words", type=int, default=1000)
parser.add_argument("--context-size", type=int, default=None, help="ignored")
parser.add_argument("--output-name", required=True)
args = parser.parse_args()

text = load_document(args.file)
title = title_of(text)
print("Title:", title)
output_dir = Path("outputs") / "method_outputs" / args.output_name
output_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()
prompt = prompt_env().get_template("title_only.j2").render(title=title, summary_length=args.summary_words)
draft = generate_summary(prompt, purpose="title_only")
final, info = finalize_summary(draft, args.summary_words,
                               compress_call=lambda p: generate_summary(p, purpose="final_compress"))
path = output_dir / f"{args.output_name}_title_only_summary_{args.summary_words}_words.txt"
path.write_text(final, encoding="utf-8")
write_metadata(output_dir, "title_only", args.file, text, start_time, args.summary_words,
               count_words(final), 0, None, extra={"title": title, "final_length_control": info})
print_run_summary("title_only", count_words(final), start_time)
