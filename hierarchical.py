from jinja2 import Environment, FileSystemLoader
from src.llm import generate_summary
import argparse

from src.document_loader import load_document
from src.chunking import chunk_document


# ============================================================
# GET DOCUMENT PATH FROM TERMINAL
# ============================================================

parser = argparse.ArgumentParser(
    description="Long Document Summarization"
)

parser.add_argument(
    "file",
    help="Path to the document file"
)

args = parser.parse_args()


# ============================================================
# LOAD DOCUMENT
# ============================================================

document_text = load_document(
    args.file
)


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = chunk_document(
    document_text,
    max_tokens=2000
)

print("Total chunks:", len(chunks))


# ============================================================
# TEST ONLY FIRST 4 CHUNKS
# ============================================================

test_chunks = chunks[:4]


# ============================================================
# LOAD TEMPLATES
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)

summarize_template = env.get_template(
    "summarize.j2"
)

extract_support_template = env.get_template(
    "extract_support.j2"
)

merge_support_template = env.get_template(
    "merge_support.j2"
)

verify_claims_template = env.get_template(
    "verify_claims.j2"
)

repair_template = env.get_template(
    "repair_verified_merge.j2"
)


# ============================================================
# SUMMARIZE EACH CHUNK
# ============================================================

chunk_summaries = []

for i, chunk in enumerate(
    test_chunks,
    start=1
):

    print(
        f"\nSummarizing chunk {i}/{len(test_chunks)}..."
    )

    prompt = summarize_template.render(
        chunk=chunk
    )

    summary = generate_summary(
        prompt
    )

    chunk_summaries.append(
        summary
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"SUMMARY OF CHUNK {i}"
    )

    print(
        "=" * 80
    )

    print(
        summary
    )


# ============================================================
# VERIFIED MERGE FUNCTION
# ============================================================

def verified_merge(
    summary_a,
    summary_b,
    source_a,
    source_b,
    label,
    max_passages=12
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"STARTING {label}"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Combine original sources
    # --------------------------------------------------------

    combined_source = (
        source_a
        + "\n\n"
        + source_b
    )


    # --------------------------------------------------------
    # Combine summaries for extraction guidance
    # --------------------------------------------------------

    combined_summaries = (
        summary_a
        + "\n\n"
        + summary_b
    )


    # --------------------------------------------------------
    # Extract support
    # --------------------------------------------------------

    extract_prompt = extract_support_template.render(
        source_text=combined_source,
        summaries=combined_summaries,
        max_passages=max_passages
    )

    support = generate_summary(
        extract_prompt
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"EXTRACTED SUPPORT - {label}"
    )

    print(
        "=" * 80
    )

    print(
        support
    )


    # --------------------------------------------------------
    # Grounded merge
    # --------------------------------------------------------

    merge_prompt = merge_support_template.render(
        summary_a=summary_a,
        summary_b=summary_b,
        support_text=support
    )

    grounded_merge = generate_summary(
        merge_prompt
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"GROUNDED MERGE - {label}"
    )

    print(
        "=" * 80
    )

    print(
        grounded_merge
    )


    # --------------------------------------------------------
    # Verify claims
    # --------------------------------------------------------

    verify_prompt = verify_claims_template.render(
        source_text=support,
        summary=grounded_merge
    )

    verification_result = generate_summary(
        verify_prompt
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"VERIFICATION - {label}"
    )

    print(
        "=" * 80
    )

    print(
        verification_result
    )


    # --------------------------------------------------------
    # Repair merge
    # --------------------------------------------------------

    repair_prompt = repair_template.render(
        summary=grounded_merge,
        verification_result=verification_result,
        support_text=support
    )

    repaired_merge = generate_summary(
        repair_prompt
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"VERIFIED MERGE - {label}"
    )

    print(
        "=" * 80
    )

    print(
        repaired_merge
    )


    return repaired_merge, combined_source


# ============================================================
# SAFETY CHECK
# ============================================================

if len(test_chunks) < 4:
    raise ValueError(
        "The document must produce at least 4 chunks for this test."
    )


# ============================================================
# VERIFIED MERGE A
# ============================================================

verified_merge_a, source_merge_a = verified_merge(
    summary_a=chunk_summaries[0],
    summary_b=chunk_summaries[1],
    source_a=test_chunks[0],
    source_b=test_chunks[1],
    label="MERGE A",
    max_passages=12
)


# ============================================================
# VERIFIED MERGE B
# ============================================================

verified_merge_b, source_merge_b = verified_merge(
    summary_a=chunk_summaries[2],
    summary_b=chunk_summaries[3],
    source_a=test_chunks[2],
    source_b=test_chunks[3],
    label="MERGE B",
    max_passages=12
)


# ============================================================
# VERIFIED FINAL MERGE
# ============================================================

verified_final_summary, final_source = verified_merge(
    summary_a=verified_merge_a,
    summary_b=verified_merge_b,
    source_a=source_merge_a,
    source_b=source_merge_b,
    label="FINAL MERGE",
    max_passages=24
)


# PRINT FINAL RESULT

print(
    "\n" + "=" * 80
)

print(
    "FINAL VERIFIED HIERARCHICAL SUMMARY"
)

print(
    "=" * 80
)

print(
    verified_final_summary
)