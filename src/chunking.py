import re
import tiktoken


encoding = tiktoken.get_encoding(
    "cl100k_base"
)


def count_tokens(text: str) -> int:

    return len(
        encoding.encode(text)
    )


def split_long_paragraph(
    paragraph: str,
    max_tokens: int
):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        paragraph
    )

    chunks = []

    current_chunk = []

    for sentence in sentences:

        candidate = " ".join(
            current_chunk + [sentence]
        )

        if count_tokens(candidate) <= max_tokens:

            current_chunk.append(
                sentence
            )

        else:

            if current_chunk:

                chunks.append(
                    " ".join(current_chunk)
                )

            # Sentence itself may still be too large
            if count_tokens(sentence) > max_tokens:

                tokens = encoding.encode(
                    sentence
                )

                for i in range(
                    0,
                    len(tokens),
                    max_tokens
                ):

                    token_chunk = tokens[
                        i:i + max_tokens
                    ]

                    chunks.append(
                        encoding.decode(
                            token_chunk
                        )
                    )

                current_chunk = []

            else:

                current_chunk = [
                    sentence
                ]

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


def chunk_document(
    text: str,
    max_tokens: int = 2000
):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []

    current_chunk = []

    for paragraph in paragraphs:

        # Very large paragraph
        if count_tokens(paragraph) > max_tokens:

            if current_chunk:

                chunks.append(
                    "\n\n".join(
                        current_chunk
                    )
                )

                current_chunk = []

            smaller_chunks = split_long_paragraph(
                paragraph,
                max_tokens
            )

            chunks.extend(
                smaller_chunks
            )

            continue

        candidate = "\n\n".join(
            current_chunk
            + [paragraph]
        )

        if count_tokens(candidate) <= max_tokens:

            current_chunk.append(
                paragraph
            )

        else:

            if current_chunk:

                chunks.append(
                    "\n\n".join(
                        current_chunk
                    )
                )

            current_chunk = [
                paragraph
            ]

    if current_chunk:

        chunks.append(
            "\n\n".join(
                current_chunk
            )
        )

    return chunks