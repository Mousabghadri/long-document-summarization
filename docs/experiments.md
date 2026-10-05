# Experiments

## GDPR

The first prototype was tested on the GDPR.

The document was split into chunks of approximately 2000 tokens.

The first 4 chunks were used during development.

A coverage problem was observed during the final hierarchical merge.

To improve this, the number of support passages in the final merge was increased.

## EU AI Act

The pipeline was later tested on the EU AI Act PDF.

The document was successfully loaded and divided into 87 chunks.

The first 4 chunks were processed with the same pipeline.

The pipeline worked successfully on a document different from the GDPR.

A remaining issue is coverage loss during the final merge.

## Sequential Summarization Parameters

The sequential summarization pipeline now supports configurable parameters:

- summary size
- context size

The summary size controls the target length of intermediate and final summaries.

The context size controls how many source tokens are processed at a time.

Example experiments on the EU AI Act:

| Context Size | Summary Size | Number of Chunks |
|---:|---:|---:|
| 2000 | 500 | 87 |
| 4000 | 500 | 35 |
| 8000 | 500 | 16 |

The summary length is controlled using prompt instructions, compression, and sentence-safe truncation when necessary.