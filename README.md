# Long Document Summarization

This project explores different approaches for summarizing long documents using Large Language Models.

The current experiments focus on legal and regulatory documents, especially:

- EU AI Act
- GDPR

However, the pipeline is designed to work with general long documents rather than being specific to legal texts.

## Methods

### 1. Sequential Summarization

The document is split into chunks.

The first chunk is summarized, then each following chunk is used to update the current running summary.

If the running summary becomes too long, it is compressed.

This method produces a coherent final summary, but each step depends on the previous step, which makes parallelization difficult.

### 2. Key Sentence Extraction

Important sentences are extracted directly from each source chunk.

The extracted sentences are validated against the original source text using exact matching after whitespace normalization.

Only sentences that can be found in the source are retained.

This provides strong grounding because the accepted output remains directly tied to the original document.

### 3. Claim Extraction and Verification

The model extracts short factual claims from each chunk.

A second verification step checks each claim against the corresponding source chunk and assigns one label:

- supported
- partial
- unsupported
- contradicted

Only supported claims are retained in the final full-document output.

The verifier preserves each original claim and does not split claims into additional atomic claims.

## Document Loading

The pipeline currently supports:

- TXT
- PDF
- DOCX

## Chunking

Documents are divided into chunks according to a configurable token context size.

Example context sizes used during experiments:

- 4000 tokens
- 8000 tokens

Larger context sizes reduce the number of chunks and may improve runtime for sequential summarization.

## Summary Lengths

Experiments were performed with approximate target lengths of:

- 250 words
- 500 words
- 1000 words
- 2000 words

For Sequential Summarization, a long final summary is generated first and shorter versions are created through additional compression steps.

For Key Sentences and Claims, shorter outputs are created by selecting complete validated sentences or supported claims without rewriting them.

## Project Structure

```text
.
├── main.py
├── hierarchical.py
├── claims.py
├── key_sentences.py
├── evaluate_outputs.py
├── quality_evaluation.py
├── comparison_results.py
│
├── scripts/
│   ├── claims_full.py
│   ├── key_sentences_full.py
│   ├── generate_length_summaries.py
│   ├── compress_summary_lengths.py
│   └── create_method_lengths.py
│
├── src/
│   ├── chunking.py
│   ├── document_loader.py
│   └── llm.py
│
├── prompts/
├── docs/
├── outputs/
└── deliverables/
```

## Running the Scripts

Scripts inside the `scripts` directory should be run as Python modules from the project root.

### Key Sentence Extraction

```bash
python -m scripts.key_sentences_full data/ai_act.pdf --context-size 8000 --max-sentences 10 --output-name ai_act
```

### Claim Extraction and Verification

```bash
python -m scripts.claims_full data/ai_act.pdf --context-size 8000 --max-claims 10 --output-name ai_act
```

### Sequential Full-Document Summary

```bash
python -m scripts.generate_length_summaries data/ai_act.pdf --summary-words 2000 --context-size 8000 --output-name ai_act
```

### Create Shorter Sequential Summaries

```bash
python -m scripts.compress_summary_lengths outputs/full_summaries/ai_act/ai_act_2000_words.txt --output-prefix ai_act
```

### Create Different Lengths for Key Sentences or Claims

Example for Key Sentences:

```bash
python -m scripts.create_method_lengths outputs/method_outputs/ai_act/ai_act_key_sentences.txt --output-prefix ai_act_key_sentences
```

Example for Claims:

```bash
python -m scripts.create_method_lengths outputs/method_outputs/ai_act/ai_act_claims.txt --output-prefix ai_act_claims
```

## Initial Evaluation

The initial evaluation compared:

- Faithfulness
- Coverage
- Coherence
- Runtime
- Number of model calls

The first small-scale experiment was performed on the first three chunks of the EU AI Act.

Further full-document experiments were then performed on both the EU AI Act and GDPR.

### Initial Quality Results

| Approach | Faithfulness | Coverage | Coherence |
|---|---:|---:|---:|
| Sequential | 5/5 | 3/5 | 5/5 |
| Key Sentences | 5/5 | 2/5 | 3/5 |
| Claims | 5/5 | 3/5 | 5/5 |

These results are preliminary and were obtained from the initial small-scale experiment.

## Full-Document Experiments

For full-document experiments, a context size of 8000 tokens was used.

### AI Act

- Sequential:
  - 16 chunks
  - approximately 2000-word final summary
  - additional 1000, 500, and 250-word versions

- Key Sentences:
  - 16 chunks
  - 115 validated sentences retained from 160 extracted sentences

- Claims:
  - 160 extracted claims
  - 142 supported
  - 18 partial
  - 0 unsupported
  - 0 contradicted

### GDPR

- Sequential:
  - 9 chunks
  - approximately 2000-word final summary
  - additional 1000, 500, and 250-word versions

- Key Sentences:
  - 86 validated sentences retained from 89 extracted sentences

- Claims:
  - 90 extracted claims
  - 88 supported
  - 2 partial
  - 0 unsupported
  - 0 contradicted

## Runtime Observations

Sequential summarization is difficult to parallelize because each step depends on the previous running summary.

Increasing the context size reduces the number of chunks and therefore reduces the number of sequential processing steps.

Key Sentence Extraction can be processed independently across chunks.

Claim Extraction can also be partially parallelized across chunks, with each chunk requiring extraction followed by verification.

Runtime is not determined only by the number of model calls. Prompt size, output length, model processing time, and API variability also affect total runtime.

## Model

Current experiments use:

`openai/gpt-oss-120b`

through OpenRouter.

## Current Status

Implemented:

- Configurable chunk/context size
- Configurable summary length
- Sequential Summarization
- Key Sentence Extraction
- Exact source validation
- Claim Extraction
- Claim Verification
- Full-document processing
- Intermediate output storage
- Runtime measurement
- Model-call measurement
- Multiple output lengths
- Initial quality evaluation
- Full-document outputs for the AI Act and GDPR

Current work focuses on comparing the approaches on complete documents and studying runtime, scalability, grounding, and coverage.