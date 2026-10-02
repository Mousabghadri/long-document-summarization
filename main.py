from jinja2 import Environment, FileSystemLoader

from src.chunking import (
    chunk_legal_document,
    count_tokens
)

from src.llm import generate_summary


# ============================================================
# 1. READ GDPR
# ============================================================

with open(
    "data/gdpr.txt",
    "r",
    encoding="utf-8"
) as f:

    gdpr_text = f.read()


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

chunks = chunk_legal_document(
    gdpr_text,
    max_tokens=2000
)

print(
    "Total chunks:",
    len(chunks)
)


# ============================================================
# 3. TEST ONLY FIRST 5 CHUNKS
# ============================================================

test_chunks = chunks[:5]


# ============================================================
# 4. LOAD JINJA2 TEMPLATES
# ============================================================

env = Environment(
    loader=FileSystemLoader("prompts")
)


summarize_template = env.get_template(
    "summarize.j2"
)

update_template = env.get_template(
    "update.j2"
)

compress_template = env.get_template(
    "compress.j2"
)

verify_claims_template = env.get_template(
    "verify_claims.j2"
)


# ============================================================
# 5. SETTINGS
# ============================================================

MAX_SUMMARY_TOKENS = 2500


# ============================================================
# 6. SUMMARIZE FIRST CHUNK
# ============================================================

first_prompt = summarize_template.render(
    chunk=test_chunks[0]
)


current_summary = generate_summary(
    first_prompt
)


print(
    "\n" + "=" * 80
)

print(
    "SUMMARY AFTER CHUNK 1"
)

print(
    "=" * 80
)

print(
    current_summary
)


# ============================================================
# 7. INCREMENTAL UPDATING
# ============================================================

for i, chunk in enumerate(
    test_chunks[1:],
    start=2
):

    print(
        f"\nProcessing chunk {i}/{len(test_chunks)}..."
    )

    update_prompt = update_template.render(

        current_summary=current_summary,

        chunk=chunk

    )


    current_summary = generate_summary(
        update_prompt
    )


    summary_tokens = count_tokens(
        current_summary
    )


    print(
        "Current summary tokens:",
        summary_tokens
    )


    # ========================================================
    # 8. COMPRESSION
    # ========================================================

    if summary_tokens > MAX_SUMMARY_TOKENS:

        print(
            "Summary too long -> compressing..."
        )


        compress_prompt = compress_template.render(
            summary=current_summary
        )


        current_summary = generate_summary(
            compress_prompt
        )


        print(
            "Tokens after compression:",
            count_tokens(current_summary)
        )


# ============================================================
# 9. PRINT SUMMARY BEFORE VERIFICATION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "SUMMARY BEFORE VERIFICATION"
)

print(
    "=" * 80
)

print(
    current_summary
)


# ============================================================
# 10. BUILD ORIGINAL SOURCE TEXT
# ============================================================

source_text = "\n\n".join(
    test_chunks
)


# ============================================================
# 11. CLAIM-BY-CLAIM VERIFICATION
# ============================================================

print(
    "\nVerifying claims against source text..."
)


verify_prompt = verify_claims_template.render(

    source_text=source_text,

    summary=current_summary

)


verification_result = generate_summary(
    verify_prompt
)


# ============================================================
# 12. PRINT VERIFICATION RESULT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CLAIM VERIFICATION RESULT"
)

print(
    "=" * 80
)

print(
    verification_result
)