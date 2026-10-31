import argparse
import sys
import time

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document
from src.chunking import count_tokens, truncate_to_tokens
from src.llm import generate_summary
from src.postprocess import count_words, finalize_summary, prompt_env
from src.run_metadata import write_metadata, print_run_summary


parser = argparse.ArgumentParser(
    description="Direct (single-prompt) summarization"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=1000
)

parser.add_argument(
    "--max-input-tokens",
    type=int,
    default=100000,
    help="Largest prompt (tokens) the model is given; keep room for output"
)

parser.add_argument(
    "--overflow",
    choices=["truncate", "fail"],
    default="truncate"
)

parser.add_argument(
    "--context-size",
    type=int,
    default=None,
    help="Ignored (accepted for a uniform CLI)"
)

parser.add_argument(
    "--output-name",
    required=True
)

args = parser.parse_args()


document_text = load_document(args.file)

template = prompt_env().get_template("direct.j2")


overhead = count_tokens(
    template.render(
        document="",
        summary_length=args.summary_words
    )
)

doc_tokens = count_tokens(document_text)

budget = args.max_input_tokens - overhead

fits = doc_tokens <= budget


print(
    "Document tokens:", doc_tokens,
    "| budget:", budget,
    "| fits:", fits
)


if not fits and args.overflow == "fail":

    sys.exit(
        f"Document ({doc_tokens} tokens) does not fit into "
        f"{budget} tokens (--overflow fail)."
    )


used_text = (
    document_text
    if fits
    else truncate_to_tokens(document_text, budget)
)

used_tokens = count_tokens(used_text)


output_dir = Path("outputs") / "method_outputs" / args.output_name

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


start_time = time.time()


# Our token count (cl100k) can differ from the model's tokenizer; if the
# provider still rejects the prompt as too long, shrink the input by 10% and
# retry (recorded in metadata as context_shrinks).

context_shrinks = 0


while True:

    try:

        draft = generate_summary(
            template.render(
                document=used_text,
                summary_length=args.summary_words
            ),
            purpose="direct_summary"
        )

        break

    except RuntimeError as e:

        if "context length" not in str(e).lower() or context_shrinks >= 3:
            raise

        context_shrinks += 1

        fits = False

        budget = int(budget * 0.9)

        used_text = truncate_to_tokens(
            document_text,
            budget
        )

        used_tokens = count_tokens(used_text)

        print(
            f"Provider context limit hit - retrying with "
            f"{budget} input tokens"
        )


final_summary, final_info = finalize_summary(
    draft,
    args.summary_words,
    compress_call=lambda p: generate_summary(
        p,
        purpose="final_compress"
    )
)

final_words = count_words(final_summary)


summary_path = (
    output_dir
    / f"{args.output_name}_direct_summary_{args.summary_words}_words.txt"
)

summary_path.write_text(
    final_summary,
    encoding="utf-8"
)


meta_path = write_metadata(
    output_dir,
    "direct",
    args.file,
    document_text,
    start_time,
    args.summary_words,
    final_words,
    1,
    None,
    extra={
        "max_input_tokens": args.max_input_tokens,
        "fits_context": fits,
        "truncated": not fits,
        "input_tokens_used": used_tokens,
        "input_fraction_used": (
            round(used_tokens / doc_tokens, 4)
            if doc_tokens
            else 1.0
        ),
        "context_shrinks": context_shrinks,
        "final_length_control": final_info,
    }
)


print_run_summary(
    "direct",
    final_words,
    start_time
)

if not fits:

    print(
        f"WARNING: document truncated to "
        f"{used_tokens}/{doc_tokens} tokens."
    )


print("Summary saved:", summary_path)
print("Metadata saved:", meta_path)