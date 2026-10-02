# Long Document Summarization

This project implements a long-document summarization pipeline using LLMs.

The system is document-agnostic and can process TXT, PDF, and DOCX files.

## Main Pipeline

```text
Long Document
→ Chunking
→ Chunk Summaries
→ Hierarchical Merge
→ Extract Support
→ Grounded Merge
→ Claim Verification
→ Repair
→ Final Verified Summary

Implemented Methods
- Incremental / Sequential Summarization
- Hierarchical Summarization
- Source-grounded merging
- Claim verification:
  - supported
  - partial
  - unsupported
  - contradicted
- Repair of unsupported or partial claims
Usage
Example:
python hierarchical.py data/document.pdf

The pipeline was tested on:
- GDPR
- EU AI Act
Current Status
The current prototype processes the first 4 chunks for testing.
The next step is to support full-document hierarchical processing and improve coverage during final merging.
Project Structure
main.py
hierarchical.py

src/
  chunking.py
  document_loader.py
  llm.py

prompts/
  summarize.j2
  update.j2
  compress.j2
  extract_support.j2
  merge_support.j2
  verify_claims.j2
  repair_verified_merge.j2

Documents used for testing are not included in the repository.