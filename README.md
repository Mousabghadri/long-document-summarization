# Long Document Summarization

This project explores and compares multiple Large Language Model approaches for summarizing long documents.

The goal is to build a document-agnostic summarization pipeline that can work across different types of long documents rather than being tied to one specific domain.

The main experiments currently use four EU regulatory documents:

- EU Artificial Intelligence Act
- General Data Protection Regulation (GDPR)
- Data Governance Act (DGA)
- Data Act

The methods themselves are designed to remain general enough to later support other document types such as technical manuals, academic texts, books, or other long-form documents.

## Main Summarization Methods

### 1. Sequential Summarization

The document is divided into chunks.

The first chunk is summarized, and each following chunk is used to update the existing running summary.

If the running summary becomes too long, a compression step is applied.

Because every step depends on the previous summary, the method is inherently sequential and difficult to parallelize.

### 2. Key Sentence Extraction / Extract-Support

For each source chunk, the model selects important complete sentences.

The sentences must be copied exactly from the source.

Each extracted sentence is validated using exact matching after whitespace normalization.

Only validated source sentences are retained.

The validated support sentences from the complete document are then used to generate a coherent grounded summary.

This provides strong source grounding while still producing a coherent abstractive final summary.

### 3. Claim Extraction and Verification

For each source chunk, the model extracts short factual claims.

A verification step checks each extracted claim against its corresponding source chunk.

Each claim receives one of four labels:

- supported
- partial
- unsupported
- contradicted

Only supported claims are retained.

The supported claims from the full document are then used to generate a coherent grounded summary.

The current pipeline also supports document-level themes that can guide claim selection and final summary organization.

## Document-Agnostic Design

An important goal of the project is to avoid prompts that are specific to a particular document domain.

Earlier prompts contained wording specific to the AI Act, GDPR, or legal documents.

These prompts were later generalized so that the pipeline can work with other document types.

Current prompts avoid assumptions about:

- laws
- regulations
- legal actors
- AI systems
- providers or deployers
- legal article structure

Instead, they use general concepts such as:

- actors
- entities
- objects
- categories
- events
- relationships
- conditions
- exceptions
- findings
- conclusions

## Document Loading

The current loader supports:

- TXT
- PDF
- DOCX

## Chunking

Documents are split into chunks according to a configurable token limit.

A common experimental setting is 8000 tokens per chunk.

Chunk size affects both runtime and summarization behavior.

Larger chunks:

- reduce the number of model calls
- reduce the number of sequential update steps
- provide more context per model call

However, chunk-based summarization can also introduce structural bias when different parts of a document have very different roles.

For example, experiments on the Data Governance Act showed that long introductory recitals can receive disproportionate representation compared with later operative sections.

## Document Structure and Selection Bias

Manual inspection of the DGA experiments revealed several important failure modes.

### Claims and Key Sentences

Both methods select information locally from individual chunks.

If each chunk receives a similar extraction budget, a chunk containing background or introductory material can receive similar representation to a chunk containing core document content.

This can create a local-importance bias.

### Sequential

Sequential summarization is order-dependent.

Early chunks establish the initial summary, while later prompts are instructed to preserve important information already present in that summary.

This can create early-chunk anchoring where the framing introduced by the beginning of a document persists throughout later updates.

These effects are now treated as important limitations of chunk-based long-document summarization.

## Document-Level Themes

An experimental Claims variant first extracts high-level document themes from a sample distributed across the beginning, middle, and end of the document.

These themes are then used to guide claim selection across individual chunks.

Example theme categories can include:

- overall purpose
- major concepts
- major sections
- important actors
- major processes
- major relationships
- important conditions
- conclusions

The themes are used only as guidance.

Individual claims must still be explicitly supported by the source chunk.

On the DGA experiment, this approach improved the LLM-as-a-Judge overall score from 2/5 to 3/5, although standard reference metrics did not improve consistently.

For this reason, the theme-guided method is currently treated as an experimental variant rather than a replacement for the original baseline.

## Evaluation

The current evaluation combines several complementary approaches.

### 1. Standard Reference-Based Metrics

Generated summaries are compared with human-written reference summaries using:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Semantic cosine similarity

Semantic similarity currently uses:

`sentence-transformers/all-MiniLM-L6-v2`

Human reference summaries are not assumed to contain every fact from the source document.

Therefore, these metrics measure alignment with the reference rather than complete factual correctness.

### 2. Source Grounding Verification

Generated summary statements are checked against the full source document.

Each statement can be classified as:

- supported
- partial
- unsupported
- contradicted

Source grounding is useful for detecting unsupported statements, but it is not treated as proof of complete factual correctness.

Manual inspection showed that automatic aggregation can sometimes hide scope errors or contradictions distributed across different source chunks.

### 3. LLM-as-a-Judge

A separate judge compares generated summaries with the human-written reference.

The final evaluation uses a large DeepSeek model through OpenRouter.

The judge evaluates:

- coverage
- emphasis
- reference alignment
- relevance
- coherence
- overall quality

The LLM judge is treated as an additional evaluation signal rather than ground truth.

### 4. Manual Inspection

All important problematic cases are also checked manually.

Manual inspection is especially important for detecting:

- scope changes
- overgeneralization
- incorrect actor attribution
- misleading document framing
- composite statements
- hidden contradictions
- document-structure bias

## Experimental Progression

The experiments followed the progression:

1 target × 1 method

↓

1 target × 3 methods

↓

4 targets × 3 methods

The current baseline contains:

**4 target documents × 3 summarization methods = 12 summaries**

Targets:

- AI Act
- GDPR
- Data Governance Act
- Data Act

Methods:

- Sequential
- Key Sentences
- Claims

## High-Level Baseline Findings

No single method performs best on every metric.

General observations include:

- Claims is often strong on semantic similarity and source grounding.
- Sequential performs particularly well on some targets such as GDPR.
- Key Sentences performs strongly on some ROUGE comparisons.
- Results vary substantially depending on document structure.
- High source-grounding scores do not always guarantee correct framing or scope.
- Human-reference alignment and source grounding measure different properties.

The DGA experiments were especially useful for exposing document-structure sensitivity.

## Efficiency

Runtime and model-call count are tracked for each method.

### Sequential

Sequential summarization is generally the most expensive because every chunk requires a new update of the running summary.

Compression steps can add additional model calls.

An optimization was added so that when the generated summary exceeds the target length by no more than 10%, deterministic sentence-safe truncation is used instead of another LLM compression call.

In a DGA 1000-word test:

- before optimization: 15 model calls
- after optimization: 13 model calls

The optimized run maintained strong quality and received an LLM-as-a-Judge overall score of 4/5.

Runtime also decreased in that run, although runtime is affected by API and model variability and should therefore be interpreted more cautiously than model-call count.

### Claims

Claims generally requires:

- 1 extraction call per chunk
- 1 verification call per chunk
- 1 final summary call

Experimental theme-guided variants require one additional document-theme extraction call.

### Key Sentences

Key Sentence extraction generally requires fewer calls because source validation is deterministic rather than requiring a separate verification model call.

## Output Organization

Outputs are organized into separate baseline, experimental, evaluation, and archived results.

### Baseline

Contains the original 4-target × 3-method comparison.

### Experiments

Contains later method improvements and diagnostic experiments.

Examples include:

- prompt-cleaning experiments
- claim-selection experiments
- document-theme experiments
- emphasis experiments
- sequential efficiency experiments

### Evaluation

Contains the evaluation results for both baseline and experimental runs.

This includes:

- standard reference-based metrics
- source-grounding verification
- LLM-as-a-Judge results

### Archive

Contains older intermediate outputs, previous verification versions, and debugging experiments retained for reference.

## Running the Main Methods

Scripts should be run from the project root using Python module syntax.

### Claims

python -m scripts.claims_full data/ai_act.pdf --context-size 8000 --max-claims 10 --summary-words 1000 --output-name ai_act

### Key Sentences

python -m scripts.key_sentences_full data/ai_act.pdf --context-size 8000 --max-sentences 10 --summary-words 1000 --output-name ai_act

### Sequential

python -m scripts.generate_length_summaries data/ai_act.pdf --summary-words 1000 --context-size 8000 --output-name ai_act

## Human Reference Evaluation

Generated summaries are compared against human-written reference summaries.

The evaluation script computes:

- summary word count
- reference word count
- length ratio
- ROUGE-1
- ROUGE-2
- ROUGE-L
- semantic cosine similarity

Example:

python human_reference_evaluation.py data/references/ai_act_human_overview.txt outputs/baseline/full_summaries/ai_act/ai_act_1000_words.txt --method "AI Act Sequential" --output-file outputs/evaluation/baseline/ai_act_sequential_standard_metrics.json

## LLM-as-a-Judge

A separate LLM judge is used to compare generated summaries with the human-written reference summary.

The final judge evaluates:

- coverage
- emphasis
- reference alignment
- relevance
- coherence
- overall quality

The final evaluation uses:

deepseek/deepseek-v3.2

through OpenRouter.

Example:

python -m scripts.llm_judge_human_reference data/references/ai_act_human_overview.txt outputs/baseline/method_outputs/ai_act/ai_act_claims_summary_1000_words_v2.txt --model deepseek/deepseek-v3.2 --output-file outputs/evaluation/baseline/final_judge_ai_act_claims.json

The LLM judge is treated as an additional evaluation signal rather than ground truth.

## Model Configuration

Main summarization experiments currently use an OpenRouter model configured through the environment variable:

MODEL_NAME

The original baseline experiments primarily used:

openai/gpt-oss-120b

The final LLM-as-a-Judge evaluation used:

deepseek/deepseek-v3.2

API keys are loaded through environment variables and must not be committed to the repository.

## Current Status

Completed:

- three long-document summarization methods
- four target documents
- 12 baseline summaries
- human-reference evaluation
- source-grounding verification
- final DeepSeek LLM-as-a-Judge evaluation
- manual inspection
- document-agnostic prompt cleanup
- document-structure bias investigation
- experimental document-level theme guidance
- initial Sequential efficiency optimization

Current work focuses on:

- improving efficiency
- understanding document-structure sensitivity
- improving cross-domain generalization
- preserving grounding while improving global document coverage

Future work includes more robust stand-alone and decontextualized claim extraction.