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