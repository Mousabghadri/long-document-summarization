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