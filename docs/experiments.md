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

## Initial Comparison of Summarization Approaches

An initial comparison was performed on the first 3 chunks of the EU AI Act using a context size of 4000 tokens.

The following three approaches were compared:

- Sequential summarization
- Key sentence extraction
- Claim extraction with verification

### Quality Results

| Approach | Faithfulness | Coverage | Coherence |
|---|---:|---:|---:|
| Sequential | 5/5 | 3/5 | 5/5 |
| Key Sentences | 5/5 | 2/5 | 3/5 |
| Claims | 5/5 | 3/5 | 5/5 |

The sequential approach produced a coherent and faithful summary, but some information was lost because the running summary had to remain within the target size.

The key sentence approach achieved exact source matching for all validated sentences, providing strong evidence that the extracted content was directly grounded in the original document. However, coverage and coherence were lower because the output consists of selected independent sentences rather than a fully integrated summary.

The claims approach achieved strong initial results in this small experiment. All extracted claims were classified as supported by the verification stage. However, claim extraction and verification are both model-based, so these results should not be interpreted as proof that hallucinations cannot occur.

### Efficiency Results

| Approach | Runtime (seconds) | Model Calls |
|---|---:|---:|
| Sequential | 158.99 | 12 |
| Key Sentences | 282.51 | 3 |
| Claims | 214.50 | 6 |

The sequential approach required the largest number of model calls because several compression calls were needed to keep the running summary within the target size.

Runtime was not directly proportional to the number of model calls. For example, the key sentence approach required only 3 model calls but had the longest runtime in this experiment. This indicates that runtime is also affected by factors such as prompt size, generated output length, model processing time, and API response variability.

### Limitations

These results are preliminary and should not be treated as final conclusions.

Only the first 3 chunks of one document were evaluated. In addition, the three approaches do not produce exactly the same type of output. Sequential summarization produces a continuous summary, while key sentence extraction and claim extraction produce more structured collections of information. Therefore, metrics such as coherence are not fully comparable across all approaches.

The same model was also used in several stages of generation and evaluation, which may introduce evaluation bias.

Further experiments should use more chunks, additional documents, different document types or languages, and repeated runs before making stronger conclusions.