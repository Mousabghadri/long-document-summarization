# Long Legal Document Summarization

Training-free LLM strategies for summarizing long legal documents, compared
under identical conditions: same prompt guidance, length control, retries and
token accounting for every method, and every comparison judged in one
evaluation run.

Results and conclusions are in the project report (shared separately). In short:
single legal documents mostly fit in a modern context window, and a single call
with the whole document (Direct) beats every chunking or extraction strategy;
Map-Reduce is the most faithful when a document does not fit.

## Setup

```bash
pip install -r requirements.txt
pip install "datasets<4.0"        # only for preparing EUR-Lex-Sum
```

Create `.env` in the project root (never commit it):

```
OPENROUTER_API_KEY=...
MODEL_NAME=openai/gpt-oss-120b
JUDGE_PROVIDER=Phala
```

Settings read from the environment (see `src/llm.py`):

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME` | summarization model (overridden by `generate --model`) |
| `JUDGE_PROVIDER` | pins the OpenRouter provider of the judge model (keeps judge scores stable) |
| `LLM_REASONING=off` | asks the summarization model not to think (all non-GPT-OSS runs) |
| `SUMMARY_LANGUAGE` | e.g. `Greek`: every prompt asks for a summary in that language |
| `LLM_TEMPERATURE`, `LLM_SEED` | default 0 and 42 |
| `LLM_PROVIDER` | optional provider pin for the summarization model |

On Windows use `$env:NAME="value"` in PowerShell and set `PYTHONUTF8=1`
before preparing datasets.

## Methods

| Method name | Script | Idea |
| --- | --- | --- |
| `direct` | `scripts/direct.py` | whole document in one call (input truncated above `--max-input-tokens`, default 100K) |
| `sequential` | `scripts/generate_length_summaries.py` | running summary refined chunk by chunk |
| `mapreduce` | `scripts/mapreduce.py` | chunk summaries in parallel, then merged |
| `mapreduce_overlap` | `scripts/mapreduce.py --overlap-tokens` | same, chunks overlap by 5% |
| `summn` | `scripts/summn.py` | SummN-style multi-stage merging |
| `key_sentences` | `scripts/key_sentences_full.py` | copy and validate key sentences per chunk, summarize them (extract-support) |
| `key_sentences_overview` | same, overview final prompt | improved final prompt (best extract-support variant) |
| `claims` | `scripts/claims_full.py` | extract and verify claims, summarize them |
| `title_only` | `scripts/title_only.py` | memorization probe: summary from the act's title only |

Rejected variants kept for reproducibility: `key_sentences_global`,
`key_sentences_overview_v2`, `_v3`, `_s20`.

## Metrics

| Metric | Script | Needs |
| --- | --- | --- |
| ROUGE-1/2/L/Lsum | `human_reference_evaluation.py` | reference (`--language` for non-English tokenisation) |
| Key-point coverage | `scripts/keypoint_coverage.py` | reference: key points extracted once, then covered/partial/missing per summary |
| BooookScore | `scripts/booookscore_eval.py` | summary only (prompt from the paper, Appendix M.4) |
| Grounding | `scripts/verify_summary_against_source.py` | source document |
| LLM judge | `scripts/llm_judge_human_reference.py` | reference |
| Semantic similarity, BERTScore | `human_reference_evaluation.py` | reference (optional) |

## Running a comparison

```bash
python compare_methods.py generate --tag T --manifest M.jsonl --target-length reference --methods direct,mapreduce
python compare_methods.py evaluate --tag T --manifest M.jsonl --coverage --booookscore
python compare_methods.py report   --tag T          # -> outputs/comparison/T/report/results.md
```

Useful options: `--model`, `--reps`, `--parallel-jobs N` (faster; wall time
then not comparable), `--reuse-support TAG:METHOD` (Key Sentences prompt
experiments), `evaluate --skip-grounding --no-semantic --judge-repeats N
--language greek`. Re-running a command retries failures; finished work is
cached.

Fairness rules built in: a tag refuses runs made with different code or
settings; averages use only documents on which every method succeeded;
compare methods only within one evaluation run. To put runs of different tags
or models into one evaluation:

```bash
python -m scripts.merge_tags --out NEW TAG_A TAG_B=direct TAG_C:@gemma
```

## Data

| Data | How to get it |
| --- | --- |
| 4 EU regulations (AI Act 2024/1689, GDPR 2016/679, Data Governance Act 2022/868, Data Act 2023/2854) | `python -m scripts.fetch_eurlex_html` (EUR-Lex HTML converted to Markdown, writes `data/manifest_html.jsonl`) |
| References for the 4 regulations | DataPACT deliverable D4.1, sections 2.2-2.5; project-internal, not in this repository (`data/references_clean/`) |
| EUR-Lex-Sum (Aumiller et al., 2022) | `python -m scripts.prepare_datasets ...` (downloads from Hugging Face, see below) |

`data/` and `outputs/` are not in git; everything in them is produced by the
commands below.

## Reproducing the experiments

Sampling uses `--seed 13`; `data/exclude_dev.txt` lists the CELEX ids of the
4 development regulations (excluded from EUR-Lex-Sum samples).

```
32016R0679
32022R0868
32023R2854
32024R1689
```

### 1. Fair comparison on the 4 regulations (GPT-OSS 120B, 3 repetitions)

```bash
python compare_methods.py generate --tag dev --manifest data/manifest_html.jsonl --reps 3 \
    --methods direct,sequential,mapreduce,key_sentences,claims,summn
python compare_methods.py evaluate --tag dev --manifest data/manifest_html.jsonl
python compare_methods.py report --tag dev
```

(The report's stage 1 used the Official Journal PDFs; the HTML input replaces
them because PDF extraction split words.)

### 2. Improving Key Sentences (final-prompt variants)

```bash
python compare_methods.py generate --tag ks --manifest data/manifest_html.jsonl --reps 3 --methods key_sentences_overview
python compare_methods.py generate --tag ks_v3 --manifest data/manifest_html.jsonl --reps 3 \
    --methods key_sentences_overview_v3 --reuse-support ks:key_sentences_overview
python compare_methods.py generate --tag ks_s20 --manifest data/manifest_html.jsonl --reps 3 --methods key_sentences_overview_s20
python -m scripts.merge_tags --out ks_eval ks ks_v3 ks_s20
python compare_methods.py evaluate --tag ks_eval --manifest data/manifest_html.jsonl
```

### 3. Held-out test: 15 EUR-Lex-Sum documents (20K-150K tokens), two models

```bash
python -m scripts.prepare_datasets --datasets eurlexsum --n 15 --seed 13 --min-tokens 20000 \
    --max-tokens 150000 --exclude-file data/exclude_dev.txt --out-name eurlex_test15
M=data/datasets/manifest_eurlex_test15.jsonl
python compare_methods.py generate --tag t15_oss --manifest $M --target-length reference \
    --methods direct,mapreduce,key_sentences,key_sentences_overview --model openai/gpt-oss-120b
LLM_REASONING=off python compare_methods.py generate --tag t15_gemma --manifest $M --target-length reference \
    --methods direct,mapreduce,key_sentences_overview --model google/gemma-4-31b-it
python -m scripts.merge_tags --out t15 t15_oss t15_gemma:@gemma
python compare_methods.py evaluate --tag t15 --manifest $M --booookscore
```

Overlapping chunks: add `--methods mapreduce_overlap` runs and merge them in.

### 4. Very long documents: 8 documents of 100K-300K tokens

```bash
python -m scripts.prepare_datasets --datasets eurlexsum --eurlex-splits test,validation,train --n 8 --seed 13 \
    --min-tokens 100000 --max-tokens 300000 --exclude-file data/exclude_dev.txt --out-name eurlex_long
python compare_methods.py generate --tag long --manifest data/datasets/manifest_eurlex_long.jsonl \
    --target-length reference --methods direct,mapreduce,key_sentences_overview
python compare_methods.py evaluate --tag long --manifest data/datasets/manifest_eurlex_long.jsonl
```

### 5. Does Direct pick up an existing summary?

```bash
python -m scripts.contamination_check data/datasets/manifest_eurlex_test15.jsonl   # recitals vs articles, writes a no-recitals copy
python compare_methods.py generate --tag norec --manifest data/datasets/manifest_eurlex_test15_norecitals.jsonl \
    --target-length reference --methods direct
python compare_methods.py generate --tag title --manifest data/datasets/manifest_eurlex_test15.jsonl \
    --target-length reference --methods title_only
python -m scripts.merge_tags --out contam t15=direct norec:@norecitals title
python compare_methods.py evaluate --tag contam --manifest data/datasets/manifest_eurlex_test15.jsonl --skip-grounding
```

### 6. Direct across model scales: full English EUR-Lex-Sum (1,503 documents)

```bash
python -m scripts.prepare_datasets --datasets eurlexsum --eurlex-splits test,validation,train --n 2000 --seed 13 \
    --min-tokens 0 --exclude-file data/exclude_dev.txt --out-name eurlex_full
M=data/datasets/manifest_eurlex_full.jsonl
for model in google/gemma-3-12b-it google/gemma-4-26b-a4b-it google/gemma-4-31b-it deepseek/deepseek-v4-flash-20260731; do
  tag=full_$(echo $model | tr '/.' '__')
  LLM_REASONING=off python compare_methods.py generate --tag $tag --manifest $M --target-length reference \
      --methods direct --model $model --parallel-jobs 6
done
python -m scripts.merge_tags --out full_models full_google_gemma-3-12b-it:@gemma3-12b \
    full_google_gemma-4-26b-a4b-it:@gemma4-26b full_google_gemma-4-31b-it:@gemma4-31b \
    full_deepseek_deepseek-v4-flash-20260731:@dsv4-flash
python compare_methods.py evaluate --tag full_models --manifest $M --skip-grounding --no-semantic --coverage --judge-repeats 1
python -m scripts.analyze_tag full_models        # length vs target, metric/judge agreement
```

### 7. Longer inputs through language: Greek EUR-Lex-Sum (1,367 documents)

```bash
python -m scripts.language_token_ratio            # Greek tokenizes ~5x English
python -m scripts.prepare_datasets --datasets eurlexsum --eurlex-lang greek --eurlex-splits test,validation,train \
    --n 2000 --seed 13 --min-tokens 0 --exclude-file data/exclude_dev.txt --out-name eurlex_greek
M=data/datasets/manifest_eurlex_greek.jsonl
export LLM_REASONING=off SUMMARY_LANGUAGE=Greek
python compare_methods.py generate --tag greek_direct --manifest $M --target-length reference --methods direct --model google/gemma-4-31b-it
python compare_methods.py generate --tag greek_mr --manifest $M --target-length reference --methods mapreduce --model google/gemma-4-31b-it
unset LLM_REASONING SUMMARY_LANGUAGE
python -m scripts.merge_tags --out greek_eval greek_direct greek_mr
python compare_methods.py evaluate --tag greek_eval --manifest $M --skip-grounding --no-semantic --coverage --judge-repeats 1 --language greek
```

### Other checks

```bash
python -m scripts.doc_lengths data/datasets/manifest_eurlex_full.jsonl     # token statistics
python -m scripts.check_eurlex_overlap                                     # which dev regulations are in EUR-Lex-Sum
python -m scripts.probe_model --model google/gemma-4-31b-it                # provider, latency, reasoning tokens
python -m scripts.which_provider                                           # judge provider
```

## Folders

```
compare_methods.py              generate / evaluate / report driver
human_reference_evaluation.py   ROUGE, semantic similarity, BERTScore
scripts/                        methods, metrics, dataset preparation, analysis
src/                            shared code (LLM client, chunking, length control, document loading)
prompts/                        Jinja prompt templates
docs/FAIRNESS_FIXES.md          what was changed to make the comparison fair
data/, outputs/                 produced by the commands above (not in git)
```
