# Design Decisions

## Generalizing the Pipeline

The first version was focused on the GDPR.

The pipeline was changed to be document-agnostic.

Main changes:

- generic document loader
- TXT, PDF and DOCX support
- generic chunking
- removal of GDPR-specific prompts

## Source-Grounded Merging

Hierarchical merging can introduce unsupported information.

To reduce hallucination, relevant passages are extracted from the original source before each merge.

## Verification and Repair

A verification stage was added after each grounded merge.

This helps identify unsupported or partially supported claims.

A repair stage is then used to correct the merged summary.