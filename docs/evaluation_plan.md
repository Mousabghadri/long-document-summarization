# Evaluation Plan

The project currently contains three approaches:

1. Sequential Summarization
2. Key Sentence Extraction
3. Claims Extraction and Verification

## Experimental Setup

Initial experiments use:

- Document: EU AI Act
- Model: openai/gpt-oss-120b
- Context size: 4000 tokens
- Initial evaluation: first 3 chunks

## Sequential Summarization

Evaluate:

- target summary size
- actual summary size
- compression behavior
- faithfulness
- coverage
- coherence

## Key Sentence Extraction

Evaluate:

- number of extracted sentences
- number of exact matches
- number of rejected sentences
- exact-match rate

## Claims Extraction

Evaluate:

- number of extracted claims
- supported claims
- partial claims
- unsupported claims
- contradicted claims

## Later Comparison

The approaches will later be compared using common quality criteria:

- Faithfulness
- Coverage
- Coherence
- Runtime
- Number of model calls

## Initial Quality Evaluation

An initial quality evaluation was performed on the first 3 chunks of the EU AI Act using the same model and context size.

### Results

| Approach | Faithfulness | Coverage | Coherence |
|---|---:|---:|---:|
| Sequential | 5/5 | 3/5 | 5/5 |
| Key Sentences | 5/5 | 2/5 | 3/5 |
| Claims | 5/5 | 3/5 | 5/5 |

### Observations

- All three approaches achieved high faithfulness in this initial test.
- Sequential summarization produced the most coherent summary.
- Key-sentence extraction had lower coherence because the output consists of independently selected source sentences rather than a rewritten summary.
- Coverage remains a limitation, especially for the key-sentence approach.
- The claims approach performed well in this small experiment, but further testing is needed because generated claims can introduce hallucinations.

### Limitations

This is only an initial evaluation using the first 3 chunks.

The results should not yet be considered final because:

- the evaluation sample is small
- the same model is used in the pipeline and evaluation
- more documents and model configurations should be tested