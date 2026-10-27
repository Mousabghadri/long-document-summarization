# Fairness fixes to the method comparison

This file lists every problem found that made the comparison between the
summarization strategies unfair (or not matching the paper's protocol), and
what was changed.  **All results produced before these fixes (including
`docs/RESULTS.md` and `outputs/`) were generated under unequal conditions and
must be regenerated** with `compare_methods.py` before they go into the paper.

## A. Unequal generation conditions

| # | Problem | Fix |
|---|---|---|
| A1 | SummN called the model with `reasoning_effort="low"` and a `max_tokens` cap (3x words); all other methods used the provider defaults (gpt-oss: medium reasoning, no cap). | All model settings come from one place, `src/llm.py` (`LLM_*` env vars). No script sets its own reasoning effort or token cap. |
| A2 | Temperature / seed were never set, so every method ran at the provider default (sampling noise differs per run). | `LLM_TEMPERATURE` (default 0) and `LLM_SEED` (default 42) are sent with every request. |
| A3 | OpenRouter may route the same model to different providers (different quantisation / limits) between calls. | Optional `LLM_PROVIDER` pins the provider and disables fallbacks (only for the summarization model; `JUDGE_PROVIDER` for the judge). |
| A4 | Retry policies differed: Sequential 3 tries, Key Sentences 3 tries but none for the final call, Claims no retry for extraction/final call (one malformed JSON killed the whole run), SummN 3 tries with growing token cap. A method that fails more often loses documents. | One retry policy in `generate_summary()` for everybody; tolerant JSON parsing (` ```json ` fences) + one re-ask via `generate_json()`. |
| A5 | Call counting differed: SummN counted only successful calls, Key Sentences counted failed ones, Sequential skipped failed ones. Tokens were not recorded at all (the paper requires them). | Every API request is counted centrally (incl. failures), with prompt / completion / reasoning tokens and summed latency, written to a uniform `metadata.json` by every method. |
| A6 | `--additional-instructions` / `--use-themes` exist only for some methods. | Never passed by `compare_methods.py`; recorded in metadata if used manually. |

## B. Unequal prompts

| # | Problem | Fix |
|---|---|---|
| B1 | The Claims final prompt contained long coverage / emphasis instructions ("start with the overall purpose", "do not let procedural details dominate", "balance major themes" ...) and strict scope/actor rules that no other method received, while the LLM judge scores exactly *coverage* and *emphasis*. | One shared block `prompts/_final_guidance.j2` (length, content & emphasis, faithfulness, format) is included in **every** prompt that writes a summary: `direct`, `summarize`, `update`, `summn_merge`, `summarize_from_support`, `summarize_from_claims`. Methods now differ only in what input the final step receives. |
| B2 | Output format differed: Claims had to write continuous prose, the others were allowed section titles and numbered lists (affects judge coherence and grounding statement splitting). | Same format rules for all (`_output_format.j2`): plain prose paragraphs. |
| B3 | Length wording differed: Key Sentences "approximately N words", Claims/SummN "at most N", Sequential update "approximately N **tokens**" while being given a word count (1000 words -> asked for ~750 words). | Same wording everywhere: at most N words, aim for 90-100 % of N. Units are words everywhere. |

## C. Unequal length control / post-processing

| # | Problem | Fix |
|---|---|---|
| C1 | Key Sentences never compressed: an over-long draft was simply cut, so it lost the end of its summary. Claims did one LLM compression, Sequential up to two, SummN only truncation. | `finalize_summary()` in `src/postprocess.py` is applied to the final output of every method: clean -> one compression call if > 110 % -> sentence-safe truncation. |
| C2 | SummN / Map-Reduce on a document that fits in ONE chunk returned the 500-word stage-1 summary as the "1000-word" summary. | Single-chunk documents are summarized directly at the final length. |
| C3 | Cleaning differed: Sequential output was never cleaned of Markdown, SummN only partly. Sequential's truncation split on ". " and destroyed paragraphs/lists. | One `clean_plain_text()` and one `truncate_to_word_limit()` (legal-abbreviation-aware) for all methods. |
| C4 | Claims: the verifier's returned claim text was used, so the verifier could re-word or add claims; the 10-claim / 10-sentence budget was not enforced. | Labels are mapped back onto the extracted claims; budgets enforced deterministically for both extraction methods. |

## D. Input problems

| # | Problem | Fix |
|---|---|---|
| D1 | `data/dga.txt` is an **empty file**, but it was the DGA input in `compare_methods.py`. | DGA now points to `data/CELEX_32022R0868_EN_TXT.pdf` (Regulation (EU) 2022/868); empty files are rejected in preflight. |
| D2 | `pypdf` splits many words of the Official Journal PDFs ("r ights", "Uni on", "REGUL A TION"). That corrupted 3 of 4 inputs and especially penalised Key Sentences, whose exact-copy validation fails when the model writes the word correctly. | PDFs are read with `pypdfium2` (clean text), `pypdf` as fallback. Key-sentence validation ignores whitespace and quote/dash variants (characters must still match exactly). |

## E. Biased evaluation

| # | Problem | Fix |
|---|---|---|
| E1 | Semantic similarity used `all-MiniLM-L6-v2` on the whole text; the model truncates at 256 word pieces, so only the first ~150-200 words of each summary and of the reference were compared. Methods that open with an overview (the Claims prompt demanded that) were rewarded. | Sentence-level embeddings, length-weighted mean; nothing is cut off. Optional BERTScore (`--bertscore`, long-context model). |
| E2 | The grounding verifier removed a hard-coded list of **GDPR** headings only; headings of other documents/methods were counted as statements and labelled unsupported. | Generic removal of heading-like lines; same splitting for every method. |
| E3 | ROUGE-L on the whole text only; no recall although summaries are 1-3x longer than references. | Adds ROUGE-Lsum (standard for multi-sentence summaries) and precision/recall; both texts cleaned identically before scoring. |
| E4 | Judge not deterministic. | Judge also runs at temperature 0 and is repeated (`--judge-repeats`), the spread is reported. |

## F. Driver / protocol

* Methods of the paper: **Direct** (`scripts/direct.py`) and **Map-Reduce** (`scripts/mapreduce.py`) were missing and are added.  Direct records whether the document fit the context; if not, the truncation is explicit and flagged in the report.
* Map-Reduce and SummN share chunking, stage-1 prompt and budget, and merge prompt (`src/hierarchical.py`) and differ only in stage organisation (one global reduce vs. fixed-size multi-stage merging), as the paper requires.
* `compare_methods.py` stores model settings and a hash of prompts+code with every run and **refuses to mix runs** with different settings/code in one tag; the report flags any mismatch.
* Averages use only documents where **every** method succeeded (no method is averaged over an easier subset).
* Efficiency table: calls (incl. retries), failed requests, prompt/completion/reasoning tokens, summed LLM latency (parallelism-independent) and wall time.
* `--target-length reference` sets each document's target to its reference length (the usual protocol for CaseSumm / IN-Abs / EUR-Lex-Sum, and needed for the compression-ratio analysis).
* Length-bin table (0-8K, 8-16K, 16-32K, 32-64K, 64K+ tokens) and per-dataset tables.
* `scripts/prepare_datasets.py` builds samples + manifest for CaseSumm, EUR-Lex-Sum (English test split) and IN-Abs (test split) with a fixed seed and an exclusion list for prompt-development documents.

## Still open (not code bugs, but needed for the paper)

1. The current experiments use 4 hand-picked EU regulations with self-made references; the paper promises CaseSumm, IN-Abs and EUR-Lex-Sum.  Use `prepare_datasets.py` + `--manifest`.
2. The paper asks for at least two summarization models: run `generate --model X --tag modelX` per model.
3. The per-chunk extraction budget (10 claims / 10 sentences per chunk) gives the extraction methods far less intermediate text than the 500-word chunk summaries of the hierarchical methods.  This is a property of the methods, but report the `support_words` stored in metadata and consider a budget ablation.
4. `--reps 3` is recommended; with temperature 0 most providers are still not fully deterministic.
5. Human evaluation with legally trained annotators (paper section 2.4).

## How to rerun

```bash
pip install -r requirements.txt
# .env: OPENROUTER_API_KEY, MODEL_NAME (+ optional LLM_PROVIDER, LLM_REASONING_EFFORT)

python compare_methods.py generate --tag eu_v2 --reps 3
python compare_methods.py evaluate --tag eu_v2 --judge-repeats 3
python compare_methods.py report   --tag eu_v2
```
