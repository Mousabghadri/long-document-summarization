# Experimental Results

This document summarizes the main results of the long-document summarization experiments.

The baseline comparison includes:

- 4 target documents
- 3 summarization methods
- 12 final summaries

Target documents:

- AI Act
- GDPR
- Data Governance Act (DGA)
- Data Act

Methods:

- Sequential Summarization
- Key Sentence Extraction / Extract-Support
- Claim Extraction and Verification

The evaluation combines:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Semantic similarity
- Source-grounding verification
- LLM-as-a-Judge
- Manual inspection

The main baseline summaries were generated at approximately 1000 words so that the methods could be compared at a similar output length.

---

## AI Act

Human reference length: 315 words.

### Standard Metrics

| Method | Words | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic Similarity |
|---|---:|---:|---:|---:|---:|
| Sequential | 963 | 0.3296 | 0.0851 | 0.1372 | 0.8008 |
| Key Sentences | 999 | 0.3166 | 0.0844 | 0.1423 | 0.6579 |
| Claims | 958 | 0.3482 | 0.1177 | 0.1522 | 0.8139 |

### Source Grounding

- Sequential: 100%
- Key Sentences: 97.44%
- Claims: final verification found no problematic statements in the retained Claims output

The Claims grounding result is therefore reported conservatively rather than assigning an exact percentage that was not consistently stored in the same final evaluation format.

### LLM-as-a-Judge

| Method | Coverage | Emphasis | Alignment | Relevance | Coherence | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 4 | 2 | 3 | 3 | 4 | 3 |
| Key Sentences | 4 | 4 | 4 | 5 | 5 | 4 |
| Claims | 4 | 4 | 4 | 5 | 5 | 4 |

### Main Observation

Claims achieved the strongest standard reference-based scores for the AI Act, including the highest ROUGE-1, ROUGE-2, ROUGE-L, and semantic similarity.

Key Sentences and Claims both received an overall LLM-as-a-Judge score of 4/5.

Sequential achieved very strong source grounding and semantic similarity, but its emphasis and reference alignment were weaker according to the final judge.

The result demonstrates that strong grounding does not automatically imply strong reference alignment.

---

## GDPR

Human reference length: 963 words.

### Standard Metrics

| Method | Words | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic Similarity |
|---|---:|---:|---:|---:|---:|
| Sequential | 988 | 0.5654 | 0.1873 | 0.2403 | 0.7941 |
| Key Sentences | 994 | 0.5305 | 0.1403 | 0.1732 | 0.6060 |
| Claims | 994 | 0.5488 | 0.1740 | 0.1860 | 0.8692 |

### Source Grounding

- Sequential: 92.86%
- Key Sentences: 93.02%
- Claims: 97.67%

The final GDPR Claims verification contained:

- 42 supported statements
- 1 partial statement
- 0 unsupported statements
- 0 contradicted statements

### LLM-as-a-Judge

| Method | Coverage | Emphasis | Alignment | Relevance | Coherence | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 5 | 5 | 4 | 5 | 4 | 5 |
| Key Sentences | 5 | 4 | 3 | 5 | 5 | 4 |
| Claims | 4 | 3 | 3 | 4 | 4 | 4 |

### Main Observation

GDPR was the strongest target for Sequential Summarization.

Sequential achieved the highest overall LLM-as-a-Judge score in the complete baseline: 5/5.

It also achieved the highest ROUGE-1, ROUGE-2, and ROUGE-L values among the three GDPR methods.

Claims achieved the highest semantic similarity at 0.8692 and the strongest source-grounding score.

The GDPR results show that different evaluation dimensions can favor different methods.

Sequential aligned most strongly with the structure and emphasis of the human reference, while Claims showed the strongest semantic similarity and grounding.

---

## Data Governance Act

Human reference length: 618 words.

### Standard Metrics

| Method | Words | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic Similarity |
|---|---:|---:|---:|---:|---:|
| Sequential | 972 | 0.4534 | 0.0818 | 0.1560 | 0.7233 |
| Key Sentences | 988 | 0.5078 | 0.1346 | 0.1753 | 0.8040 |
| Claims | 962 | 0.4939 | 0.1202 | 0.1777 | 0.7950 |

### Source Grounding

- Sequential: 97.92%
- Key Sentences: 89.47%
- Claims: 92.31%

These percentages must be interpreted carefully.

Manual inspection identified problems that were not fully represented by the aggregate grounding percentage.

The Sequential summary contained an overbroad statement suggesting unrestricted transfers of non-personal data.

The Claims summary contained a temporal factual error.

Therefore, high automatic grounding scores should not be treated as proof of factual correctness.

### LLM-as-a-Judge

| Method | Coverage | Emphasis | Alignment | Relevance | Coherence | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 3 | 2 | 2 | 3 | 4 | 3 |
| Key Sentences | 2 | 2 | 1 | 3 | 5 | 2 |
| Claims | 2 | 2 | 2 | 3 | 5 | 2 |

### Main Observation

The DGA was the most useful document for exposing weaknesses in the chunk-based methods.

Key Sentences achieved the strongest ROUGE-1, ROUGE-2, and semantic similarity values.

Claims achieved the highest ROUGE-L.

However, both Key Sentences and Claims received only 2/5 overall from the LLM judge.

Sequential received 3/5.

Manual inspection showed that the summaries often focused too heavily on particular operational details instead of reflecting the overall document emphasis.

---

## DGA Structural Bias Investigation

The poor DGA judge results triggered a deeper investigation of document structure.

With an 8000-token context size, the DGA is divided into five chunks.

The early parts of the document contain many recitals, while the operative Articles appear mainly in later chunks.

This revealed different structural weaknesses in the three methods.

### Claims and Key Sentences

Claims and Key Sentences select information locally within individual chunks.

Each chunk receives a similar extraction budget.

As a result, a recital-heavy chunk can contribute approximately the same number of selected items as a chunk containing important operative provisions.

This creates a local-importance bias.

The extraction process can therefore preserve information that is genuinely important inside a chunk but not proportionally important to the document as a whole.

### Sequential

Sequential Summarization has a different weakness.

The first chunk establishes the initial running summary.

Later chunks update that summary while preserving information considered important from previous steps.

This creates order dependence and early-chunk anchoring.

If the initial part of the document emphasizes background material, that framing can remain influential throughout the later summarization steps.

### General Finding

Chunk-level importance is not necessarily equivalent to document-level importance.

This became one of the main findings of the experiments.

---

## Data Act

Human reference length: 517 words.

### Standard Metrics

| Method | Words | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic Similarity |
|---|---:|---:|---:|---:|---:|
| Sequential | 994 | 0.3335 | 0.0683 | 0.1453 | 0.7318 |
| Key Sentences | 964 | 0.3846 | 0.0875 | 0.1565 | 0.7120 |
| Claims | 981 | 0.3569 | 0.0758 | 0.1553 | 0.7211 |

### Source Grounding

- Sequential: 85.11%
- Key Sentences: 94.59%
- Claims: 97.22%

### LLM-as-a-Judge

| Method | Coverage | Emphasis | Alignment | Relevance | Coherence | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 4 | 3 | 3 | 4 | 4 | 4 |
| Key Sentences | 2 | 1 | 2 | 2 | 5 | 2 |
| Claims | 3 | 2 | 2 | 2 | 4 | 2 |

### Main Observation

Key Sentences achieved the highest ROUGE-1, ROUGE-2, and ROUGE-L values for the Data Act.

Claims achieved the strongest source-grounding score.

However, Sequential received the strongest LLM-as-a-Judge result at 4/5.

Key Sentences and Claims both received 2/5 overall.

This again demonstrates that sentence-level or claim-level grounding does not guarantee that the final summary has the same emphasis or global coverage as the human reference.

---

## Baseline LLM-as-a-Judge Overview

The final overall LLM-as-a-Judge scores were:

| Document | Sequential | Key Sentences | Claims |
|---|---:|---:|---:|
| AI Act | 3 | 4 | 4 |
| GDPR | 5 | 4 | 4 |
| DGA | 3 | 2 | 2 |
| Data Act | 4 | 2 | 2 |

### Observation

Sequential received the strongest overall judge result on GDPR and the Data Act.

Claims and Key Sentences performed particularly well on the AI Act.

The DGA was difficult for all three methods.

The results suggest that method performance depends strongly on document structure and reference-summary emphasis.

---

## Cross-Document Findings

No single method consistently outperformed the others across all metrics and documents.

### Sequential Summarization

Strengths:

- produces coherent summaries
- performed very strongly on GDPR
- achieved strong judge results on the Data Act
- naturally integrates information into a single narrative

Weaknesses:

- inherently sequential
- expensive in model calls and runtime
- difficult to parallelize
- sensitive to chunk order
- vulnerable to early-chunk anchoring

### Key Sentence Extraction

Strengths:

- extracted support can be validated directly against the source
- deterministic exact-source validation provides strong evidence that selected support exists
- achieved strong ROUGE results on the DGA and Data Act
- requires fewer model calls than Claims

Weaknesses:

- local chunk-level selection does not guarantee global importance
- equal extraction budgets across chunks can distort document emphasis
- validated sentences can still be used to create a globally unbalanced summary
- exact source presence does not prove that the final interpretation has correct scope

### Claim Extraction and Verification

Strengths:

- strong source grounding
- high semantic similarity on several targets
- explicit claim verification stage
- provides structured intermediate information
- supports later grounded summary generation

Weaknesses:

- requires extraction and verification model calls
- local claim selection can overrepresent some document sections
- claim verification does not automatically solve global coverage or emphasis
- composite or context-dependent claims can still create difficulties

---

## Grounding Evaluation Caveat

The source-grounding verifier evaluates summary statements against document chunks and aggregates the results.

This is useful, but it has an important limitation.

A statement may appear supported in one chunk even when another part of the document provides an exception, narrower scope, or contradictory qualification.

The current aggregation therefore should not be interpreted as formal proof that every high-scoring summary is completely correct.

This limitation was visible during manual inspection of the DGA Sequential summary.

For this reason, grounding evaluation is combined with:

- human-reference metrics
- LLM-as-a-Judge
- manual inspection

---

## Experimental Claims Improvements

After the baseline evaluation, several DGA Claims experiments were performed to investigate whether global document coverage and emphasis could be improved.

These experiments are treated as experimental variants rather than replacements for the baseline.

### Prompt-Cleaning Variant

The prompts were cleaned to remove domain-specific wording and make the pipeline more document-agnostic.

This improved the generality of the method, but prompt cleanup alone did not solve the document-level emphasis problem.

### Claim-Selection Variant

A more general claim-selection prompt was tested.

Results:

- 50 extracted claims
- 43 supported
- 7 partial
- 940-word summary

Standard metrics:

- ROUGE-1: 0.4898
- ROUGE-2: 0.1421
- ROUGE-L: 0.1789
- Semantic similarity: 0.7811

LLM-as-a-Judge:

- Coverage: 2
- Emphasis: 2
- Alignment: 2
- Relevance: 2
- Coherence: 5
- Overall: 2

The variant did not solve the main document-level problem.

---

## Theme-Guided Claims

A new experimental approach was then tested.

Before claim extraction, the model identifies high-level document themes using material sampled from the beginning, middle, and end of the document.

The extracted themes are then passed to the claim-selection process.

The goal is to help local chunk-level extraction remain aware of the broader document structure.

The extracted DGA themes included areas such as:

- overall purpose and scope
- reuse of protected public-sector data
- data intermediation services
- data altruism
- governance mechanisms
- interaction with EU law
- third-country data transfers
- enforcement

### Initial Theme-Guided Run

Results:

- 50 extracted claims
- 43 supported
- 7 partial
- 1118-word summary
- 12 model calls

Standard metrics:

- ROUGE-1: 0.4421
- ROUGE-2: 0.1104
- ROUGE-L: 0.1714
- Semantic similarity: 0.7565

LLM-as-a-Judge:

- Coverage: 3
- Emphasis: 2
- Alignment: 2
- Relevance: 3
- Coherence: 5
- Overall: 3

The overall judge score improved from the baseline Claims score of 2/5 to 3/5.

However, standard reference metrics decreased.

---

## Theme-Guided Final Summary

Document themes were also passed to the final summary-generation stage.

Additional instructions emphasized:

- the overall document purpose
- major themes before details
- balanced coverage
- representative rather than exhaustive operational details

Results:

- 50 extracted claims
- 40 supported
- 10 partial
- 1005-word summary
- 12 model calls

Standard metrics:

- ROUGE-1: 0.4558
- ROUGE-2: 0.1185
- ROUGE-L: 0.1816
- Semantic similarity: 0.7347

LLM-as-a-Judge:

- Coverage: 3
- Emphasis: 2
- Alignment: 3
- Relevance: 3
- Coherence: 5
- Overall: 3

This improved reference alignment according to the judge but again did not consistently improve the standard metrics.

---

## Additional Emphasis Experiment

A further variant added stronger instructions about global document emphasis.

Results:

- 50 extracted claims
- 50 supported
- 0 partial
- 1078-word summary
- 12 model calls

Standard metrics:

- ROUGE-1: 0.4652
- ROUGE-2: 0.1347
- ROUGE-L: 0.1771
- Semantic similarity: 0.7471

LLM-as-a-Judge:

- Coverage: 3
- Emphasis: 2
- Alignment: 2
- Relevance: 3
- Coherence: 5
- Overall: 3

The experiment did not produce a clear additional improvement.

Further prompt tuning on the DGA was stopped to avoid overfitting the method to one document or one judge.

The theme-guided approach is therefore considered promising but not yet a confirmed replacement for the baseline Claims method.

---

## Sequential Efficiency Experiment

Sequential Summarization was identified as the most expensive method because every new chunk requires updating the existing summary.

The process may also require additional LLM calls to compress the running summary when it exceeds the target length.

A small efficiency optimization was introduced.

If the current summary exceeds the target length by no more than 10%, deterministic sentence-safe truncation is used instead of another LLM compression request.

### DGA 1000-Word Test

Before optimization:

- 5 chunks
- 15 model calls
- 2059.04 seconds
- 999 words

After optimization:

- 5 chunks
- 13 model calls
- 1244.06 seconds
- 988 words

This reduced the model-call count from 15 to 13.

This corresponds to approximately a 13% reduction in model calls.

The runtime also decreased substantially in this run, but runtime is affected by model and API variability and should not be attributed entirely to the optimization.

### Quality After Optimization

Standard metrics:

- ROUGE-1: 0.4285
- ROUGE-2: 0.1109
- ROUGE-L: 0.1570
- Semantic similarity: 0.8008

LLM-as-a-Judge:

- Coverage: 4
- Emphasis: 3
- Alignment: 3
- Relevance: 4
- Coherence: 5
- Overall: 4

The optimization therefore reduced model calls without causing an obvious quality collapse in this experiment.

---

## Main Conclusions

The experiments produced several main findings.

1. No single summarization method dominates across all documents and evaluation metrics.

2. Source grounding and reference alignment measure different properties.

3. A summary can be strongly grounded in the source while still having poor document-level emphasis.

4. Local chunk-level importance is not necessarily equivalent to global document-level importance.

5. Sequential Summarization is vulnerable to order dependence and early-chunk anchoring.

6. Key Sentence and Claims methods are vulnerable to local-selection bias when extraction budgets are distributed similarly across structurally different chunks.

7. Human-reference metrics, source verification, LLM judging, and manual inspection are complementary rather than interchangeable.

8. Automatic grounding percentages should not be interpreted as complete proof of factual correctness.

9. Document-level theme guidance is a promising direction for improving global coverage, but current results are mixed.

10. Sequential efficiency can be improved by avoiding unnecessary LLM compression calls.

---

## Current Research Direction

The current work focuses on:

- improving global document coverage
- reducing structural bias
- improving efficiency
- preserving source grounding
- improving cross-domain generalization
- avoiding document-specific prompt assumptions

The next planned direction is more robust decontextualized and stand-alone claim extraction.

This is intended to improve the quality and usability of intermediate claims before continuing with broader long-document summarization experiments.