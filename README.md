# Long Legal Document Summarization

Training-free LLM strategies for summarizing long legal documents, compared
under identical conditions (same prompt guidance, length control, retries and
token accounting for every method).

Results and conclusions: `Legal_Summarization_Results.docx` (shared separately).
In short: Direct (one call with the whole document) is best when the document
fits in the context; Map-Reduce is the most faithful for longer documents;
Gemma 4 31B (non-thinking) beats GPT-OSS 120B on every method and is faster.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in the project root:

```
OPENROUTER_API_KEY=...
MODEL_NAME=openai/gpt-oss-120b
JUDGE_PROVIDER=Phala
```

Optional settings (see `src/llm.py`): `LLM_REASONING=off` (non-thinking runs),
`LLM_PROVIDER`, `LLM_TEMPERATURE` (default 0), `LLM_SEED` (default 42),
`LLM_MAX_RETRIES`.

## Methods

| Method name | Script | Idea |
| --- | --- | --- |
| `direct` | `scripts/direct.py` | whole document in one call (truncated above `--max-input-tokens`) |
| `sequential` | `scripts/generate_length_summaries.py` | running summary updated chunk by chunk |
| `mapreduce` | `scripts/mapreduce.py` | chunk summaries in parallel, then merged |
| `mapreduce_overlap` | `scripts/mapreduce.py --overlap-tokens` | same, chunks overlap by 5% |
| `summn` | `scripts/summn.py` | multi-stage merging of chunk summaries |
| `key_sentences` | `scripts/key_sentences_full.py` | copy and validate key sentences per chunk, summarize them |
| `key_sentences_overview` | same, `--summary-prompt summarize_from_support_overview.j2` | improved final prompt (best variant) |
| `claims` | `scripts/claims_full.py` | extract and verify claims, summarize them |

Other variants (`key_sentences_global`, `_overview_v2`, `_overview_v3`,
`_overview_s20`) are the rejected experiments described in the results.

## Running a comparison

Everything goes through `compare_methods.py`, in three steps:

```bash
python compare_methods.py generate --tag mytag --methods direct,mapreduce --reps 1
python compare_methods.py evaluate --tag mytag --eval-workers 6
python compare_methods.py report   --tag mytag
```

The report is written to `outputs/comparison/<tag>/report/results.md`.

Useful options:

- `--manifest file.jsonl --target-length reference`: run on a dataset; the
  target length is the length of each reference summary.
- `--model google/gemma-4-31b-it`: another summarization model.
- `--parallel-jobs 4`: generate several summaries at once (wall time is then
  not comparable; keep 1 for timing).
- `--reuse-support TAG:METHOD`: prompt experiments on Key Sentences; reuse the
  extracted sentences of an earlier run so only the final step runs.
- `evaluate --skip-grounding`, `--judge-repeats N`: faster evaluation.

Failed jobs or evaluations are retried by running the same command again;
finished work is cached.

## Rules for fair comparisons

- LLM-judge scores drift between runs. Only compare methods that were judged
  in the same `evaluate` run. To compare runs from different tags, merge their
  `runs.json` into a new tag and evaluate that tag.
- A tag refuses to mix runs made with different code or settings.
- Averages use only documents on which every method succeeded.

## Datasets

```bash
pip install "datasets<4.0"
python -m scripts.prepare_datasets --datasets eurlexsum --n 15 --min-tokens 20000 \
    --exclude-file data/exclude_dev.txt
```

Writes the documents and `data/datasets/manifest_<name>.jsonl`. Options:
`--eurlex-splits test,validation,train`, `--max-tokens`, `--out-name`.
On Windows set `PYTHONUTF8=1` first.

## Helper scripts

- `scripts/probe_model.py --model X`: provider, latency and reasoning tokens of a model.
- `scripts/which_provider.py`: which OpenRouter provider serves the judge.
- `scripts/verify_summary_against_source.py`: grounding of one summary.
- `scripts/tidy_project.py`: move unused files and tags to `archive/`.

## Folders

```
compare_methods.py      generate / evaluate / report driver
human_reference_evaluation.py   ROUGE and semantic similarity
scripts/                methods and helpers
src/                    shared code (LLM client, chunking, length control)
prompts/                Jinja prompt templates
data/                   source documents, references, manifests (not in git)
outputs/                summaries, evaluations, reports (not in git)
archive/                old scripts and experiments (not in git)
docs/FAIRNESS_FIXES.md  what was changed to make the comparison fair
```