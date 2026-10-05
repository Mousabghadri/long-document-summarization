# Progress Log

## 2026-10-05

- Added configurable summary size for sequential summarization.
- Added configurable context size.
- Tested context sizes of 2000, 4000 and 8000 tokens on the EU AI Act.
- Added compression and sentence-safe truncation for summary-length control.
- Added key-sentence extraction.
- Added exact-match validation after whitespace normalization.
- Added storage of extracted, validated and rejected key sentences.
- Added claims extraction as a separate experimental approach.
- Added claim verification using supported, partial, unsupported and contradicted labels.
- Added storage of intermediate claims and verification outputs.