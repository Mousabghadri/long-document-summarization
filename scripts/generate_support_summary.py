import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.llm import generate_summary


def count_words(text):
    return len(text.split())


def truncate_to_word_limit(text, max_words):
    sentences = text.split(". ")

    selected = []
    total_words = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        words = sentence.split()

        if total_words + len(words) > max_words:
            break

        selected.append(sentence)
        total_words += len(words)

    result = ". ".join(selected).strip()

    if result and not result.endswith("."):
        result += "."

    return result


parser = argparse.ArgumentParser(
    description="Generate summary from validated support sentences"
)

parser.add_argument(
    "support_file",
    help="Path to validated support sentences file"
)

parser.add_argument(
    "--summary-words",
    type=int,
    default=1000
)

parser.add_argument(
    "--additional-instructions",
    type=str,
    default=""
)

parser.add_argument(
    "--output-file",
    required=True
)

args = parser.parse_args()


support_text = Path(
    args.support_file
).read_text(
    encoding="utf-8"
)


env = Environment(
    loader=FileSystemLoader("prompts")
)

summary_template = env.get_template(
    "summarize_from_support.j2"
)

compress_template = env.get_template(
    "compress.j2"
)


prompt = summary_template.render(
    support_text=support_text,
    summary_length=args.summary_words,
    additional_instructions=args.additional_instructions
)


print("Generating summary...")

summary = generate_summary(prompt)

print(
    "Initial summary words:",
    count_words(summary)
)


original_summary = summary

if count_words(summary) > args.summary_words:

    print("Compressing summary...")

    compress_prompt = compress_template.render(
        summary=summary,
        target_words=args.summary_words
    )

    compressed_summary = generate_summary(
        compress_prompt
    )

    compressed_words = count_words(
        compressed_summary
    )

    print(
        "Compressed words:",
        compressed_words
    )

    minimum_acceptable = int(
        args.summary_words * 0.90
    )

    if (
        minimum_acceptable
        <= compressed_words
        <= args.summary_words
    ):
        summary = compressed_summary

    else:
        print(
            "Compression rejected because length is too far from target."
        )

        summary = original_summary


if count_words(summary) > args.summary_words:

    print(
        "Applying final word-limit fallback..."
    )

    summary = truncate_to_word_limit(
        summary,
        args.summary_words
    )


output_path = Path(
    args.output_file
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    summary,
    encoding="utf-8"
)


print(
    "Final summary words:",
    count_words(summary)
)

print(
    "Saved:",
    output_path
)