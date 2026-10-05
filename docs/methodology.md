# Methodology

## Incremental Summarization

The document is split into chunks.

The first chunk is summarized, and the running summary is updated with each new chunk.

## Hierarchical Summarization

Each chunk is summarized independently.

The summaries are merged in pairs and then merged again at higher levels.

## Extract Support

Before merging, relevant passages are extracted from the original source.

These passages are used as factual evidence.

## Claim Verification

Each generated claim is classified as:

- supported
- partial
- unsupported
- contradicted

## Repair

Supported claims are kept.

Partial claims are reduced to the supported meaning.

Unsupported and contradicted claims are removed.

## Key Sentence Extraction

A key-sentence extraction approach was added.

For each source chunk, the model selects important sentences verbatim from the source.

The extracted sentences are validated programmatically using exact matching after whitespace normalization.

Intermediate outputs are saved as:

- extracted sentences
- validated sentences
- rejected sentences

## Claims Extraction and Verification

A claims-based approach is also kept as a separate experimental method.

Claims are first extracted from the source text and then verified against the source.

Each claim is classified as:

- supported
- partial
- unsupported
- contradicted

The extracted claims and verification results are saved to disk for later analysis and comparison.