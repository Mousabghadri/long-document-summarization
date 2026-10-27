from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import threading
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Defaults - edit here if your files live elsewhere (or use --doc name=path)
# --------------------------------------------------------------------------
DEFAULT_DOCS = {
    "ai_act": "data/ai_act.pdf",
    "gdpr": "data/gdpr.txt",
    # data/dga.txt in the project is an EMPTY file; the DGA (Regulation (EU)
    # 2022/868) is the CELEX_32022R0868 PDF
    "dga": "data/CELEX_32022R0868_EN_TXT.pdf",
    "data_act": "data/OJ_L_202302854_EN_TXT.pdf",
}
REFERENCE_PATTERN = "data/references/{doc}_human_overview.txt"

METHODS = ["direct", "sequential", "mapreduce", "key_sentences", "key_sentences_global",
           "key_sentences_overview", "key_sentences_overview_v2",
           "key_sentences_overview_v3",
           "key_sentences_overview_s20",
           "mapreduce_overlap", "claims", "summn"]
LABELS = {
    "direct": "Direct",
    "sequential": "Sequential",
    "mapreduce": "Map-Reduce",
    "key_sentences": "Extract-support",
    "key_sentences_global": "Extract-support (global)",
    "key_sentences_overview": "Extract-support (overview)",
    "key_sentences_overview_v2": "Extract-support (overview v2)",
    "key_sentences_overview_v3": "Extract-support (overview v3)",
    "key_sentences_overview_s20": "Extract-support (overview, 20 sent.)",
    "mapreduce_overlap": "Map-Reduce (overlap)",
    "claims": "Claims",
    "summn": "SummN-style",
}
LENGTH_BINS = [(0, 8000, "0-8K"), (8000, 16000, "8-16K"), (16000, 32000, "16-32K"),
               (32000, 64000, "32-64K"), (64000, 10 ** 9, "64K+")]

ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"] = "1"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def base_dir(a) -> Path:
    return ROOT / "outputs" / "comparison" / a.tag


def run_name(a, doc: str, method: str, rep: int) -> str:
    return f"cmp_{safe(a.tag)}_{safe(doc)}_{method}_r{rep}"


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def count_words(text: str) -> int:
    return len(text.split())


def code_hash() -> str:
    """Hash of everything that defines a method's behaviour (prompts + code)."""
    h = hashlib.sha256()
    files = sorted((ROOT / "prompts").glob("*.j2")) + sorted((ROOT / "src").glob("*.py")) + \
        [ROOT / "scripts" / f for f in ("direct.py", "generate_length_summaries.py", "mapreduce.py",
                                        "key_sentences_full.py", "claims_full.py", "summn.py")]
    for f in files:
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def current_llm_settings(a) -> dict:
    if getattr(a, "model", None):
        os.environ["MODEL_NAME"] = a.model
    sys.path.insert(0, str(ROOT))
    from src.llm import llm_settings  # noqa: E402  (loads .env)
    return llm_settings()


def resolve_docs(a) -> dict:
    """name -> {"source": Path, "reference": Path, "dataset": str}"""
    entries = {}
    if a.manifest:
        for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            entries[d["name"]] = {"source": d["source"], "reference": d["reference"],
                                  "dataset": d.get("dataset", "manifest")}
    else:
        for name, path in DEFAULT_DOCS.items():
            entries[name] = {"source": path, "reference": REFERENCE_PATTERN.format(doc=name),
                             "dataset": "eu_regulations"}
    for item in a.doc or []:
        if "=" not in item:
            sys.exit(f"--doc expects name=path, got: {item}")
        name, path = item.split("=", 1)
        entries[name.strip()] = {"source": path.strip(),
                                 "reference": REFERENCE_PATTERN.format(doc=name.strip()),
                                 "dataset": "custom"}
    if a.docs:
        wanted = [d.strip() for d in a.docs.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in entries]
        if unknown:
            sys.exit(f"Unknown documents: {unknown}. Known: {list(entries)[:20]}...")
        entries = {d: entries[d] for d in wanted}
    out = {}
    for name, e in entries.items():
        out[name] = {"source": (ROOT / e["source"]).resolve(),
                     "reference": (ROOT / e["reference"]).resolve(),
                     "dataset": e["dataset"]}
    return out


def base_method(m: str) -> str:
    """'direct@gemma' -> 'direct' (a suffix marks runs of another model in a merged tag)."""
    return m.split("@", 1)[0]


def label(m: str) -> str:
    b, _, suffix = m.partition("@")
    return LABELS[b] + (f" [{suffix}]" if suffix else "")


def pick_methods(a) -> list:
    if not a.methods:
        return list(METHODS)
    chosen = [m.strip() for m in a.methods.split(",") if m.strip()]
    unknown = [m for m in chosen if base_method(m) not in METHODS]
    if unknown:
        sys.exit(f"Unknown methods: {unknown}. Known: {METHODS}")
    return chosen


def preflight(docs: dict, need_refs: bool) -> None:
    print(f"Documents: {len(docs)}")
    missing = []
    for name, d in docs.items():
        if not d["source"].exists():
            missing.append(str(d["source"]))
        elif d["source"].stat().st_size == 0:
            missing.append(f"{d['source']} (EMPTY FILE)")
        if need_refs and not d["reference"].exists():
            missing.append(str(d["reference"]))
    if len(docs) <= 12:
        for name, d in docs.items():
            print(f"  {name:12s} source: {'OK ' if d['source'].exists() else 'MISSING'} {d['source']}")
            if need_refs:
                print(f"  {'':12s} ref:    {'OK ' if d['reference'].exists() else 'MISSING'} {d['reference']}")
    if missing:
        sys.exit("\nMissing files:\n  " + "\n  ".join(missing[:20]) +
                 "\nFix the paths (see DEFAULT_DOCS / --doc / --manifest) and retry.")
    print()


def target_words_for(a, d: dict) -> int:
    if a.target_length == "fixed":
        return a.summary_words
    ref_words = count_words(d["reference"].read_text(encoding="utf-8"))
    return max(50, int(round(ref_words * a.reference_ratio)))


def method_order(methods: list, doc_index: int) -> list:
    k = doc_index % len(methods)
    return methods[k:] + methods[:k]


# --------------------------------------------------------------------------
# GENERATE
# --------------------------------------------------------------------------
def build_cmd(method: str, src: Path, name: str, words: int, a) -> list:
    cmd = _build_cmd(method, src, name, words, a)
    reuse = getattr(a, "reuse_support", None)
    if reuse and method.startswith("key_sentences") and method != "key_sentences_global":
        cmd += ["--support-file", str(support_file_for(reuse, a, name, method))]
    return cmd


def support_file_for(reuse: str, a, name: str, method: str) -> Path:
    """--reuse-support TAG:METHOD -> support file of the same doc/rep in that tag."""
    src_tag, src_method = reuse.split(":", 1)
    prefix = f"cmp_{safe(a.tag)}_"
    rest = name[len(prefix):]                      # <doc>_<method>_r<rep>
    marker = f"_{method}_r"
    i = rest.rfind(marker)
    doc_part, rep_part = rest[:i], rest[i + len(marker):]
    src_name = f"cmp_{safe(src_tag)}_{doc_part}_{src_method}_r{rep_part}"
    f = ROOT / "outputs" / "method_outputs" / src_name / f"{src_name}_support_sentences.txt"
    if not f.exists():
        sys.exit(f"--reuse-support: support file not found: {f}")
    return f


def _build_cmd(method: str, src: Path, name: str, words: int, a) -> list:
    py = sys.executable
    w, c, k = str(words), str(a.context_size), str(a.workers)
    if method == "direct":
        return [py, "-m", "scripts.direct", str(src), "--summary-words", w,
                "--max-input-tokens", str(a.max_input_tokens), "--overflow", a.direct_overflow,
                "--output-name", name]
    if method == "sequential":
        return [py, "-m", "scripts.generate_length_summaries", str(src),
                "--summary-words", w, "--context-size", c, "--output-name", name, "--no-resume"]
    if method == "mapreduce":
        return [py, "-m", "scripts.mapreduce", str(src), "--context-size", c, "--summary-words", w,
                "--chunk-summary-words", str(a.chunk_summary_words),
                "--intermediate-summary-words", str(a.intermediate_summary_words),
                "--max-input-tokens", str(a.max_input_tokens),
                "--workers", k, "--output-name", name]
    if method == "mapreduce_overlap":
        return [py, "-m", "scripts.mapreduce", str(src), "--context-size", c, "--summary-words", w,
                "--chunk-summary-words", str(a.chunk_summary_words),
                "--intermediate-summary-words", str(a.intermediate_summary_words),
                "--max-input-tokens", str(a.max_input_tokens),
                "--overlap-tokens", str(int(a.context_size * 0.05)),
                "--workers", k, "--output-name", name]
    if method == "key_sentences":
        return [py, "-m", "scripts.key_sentences_full", str(src),
                "--context-size", c, "--max-sentences", str(a.max_sentences),
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "key_sentences_global":
        return [py, "-m", "scripts.key_sentences_global", str(src),
                "--context-size", c, "--max-sentences", str(a.max_sentences),
                "--candidates-per-chunk", "20",
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "key_sentences_overview":
        return [py, "-m", "scripts.key_sentences_full", str(src),
                "--context-size", c, "--max-sentences", str(a.max_sentences),
                "--summary-prompt", "summarize_from_support_overview.j2",
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "key_sentences_overview_v2":
        return [py, "-m", "scripts.key_sentences_full", str(src),
                "--context-size", c, "--max-sentences", str(a.max_sentences),
                "--summary-prompt", "summarize_from_support_overview_v2.j2",
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "key_sentences_overview_s20":
        return [py, "-m", "scripts.key_sentences_full", str(src),
                "--context-size", c, "--max-sentences", "20",
                "--summary-prompt", "summarize_from_support_overview.j2",
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "key_sentences_overview_v3":
        return [py, "-m", "scripts.key_sentences_full", str(src),
                "--context-size", c, "--max-sentences", str(a.max_sentences),
                "--summary-prompt", "summarize_from_support_overview_v3.j2",
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "claims":
        return [py, "-m", "scripts.claims_full", str(src),
                "--context-size", c, "--max-claims", str(a.max_claims),
                "--summary-words", w, "--output-name", name, "--workers", k]
    if method == "summn":
        return [py, "-m", "scripts.summn", str(src),
                "--context-size", c, "--summary-words", w,
                "--chunk-summary-words", str(a.chunk_summary_words),
                "--intermediate-summary-words", str(a.intermediate_summary_words),
                "--merge-group-size", str(a.merge_group_size),
                "--workers", k, "--output-name", name]
    raise ValueError(method)


def output_folder(method: str, name: str) -> Path:
    sub = "full_summaries" if method == "sequential" else "method_outputs"
    return ROOT / "outputs" / sub / name


def expected_summary(method: str, name: str, words: int) -> Path:
    if method == "sequential":
        return output_folder(method, name) / f"{name}_{words}_words.txt"
    suffix = {
        "direct": "direct_summary",
        "mapreduce": "mapreduce_summary",
        "mapreduce_overlap": "mapreduce_summary",
        "key_sentences": "key_sentences_summary",
        "key_sentences_global": "key_sentences_global_summary",
        "key_sentences_overview": "key_sentences_summary",
        "key_sentences_overview_v2": "key_sentences_summary",
        "key_sentences_overview_v3": "key_sentences_summary",
        "key_sentences_overview_s20": "key_sentences_summary",
        "claims": "claims_summary",
        "summn": "summn_summary",
    }[method]
    return output_folder(method, name) / f"{name}_{suffix}_{words}_words.txt"


def wipe_outputs(name: str) -> None:
    for d in (ROOT / "outputs" / "method_outputs" / name,
              ROOT / "outputs" / "full_summaries" / name):
        if d.exists():
            shutil.rmtree(d)


def generate_one(method, doc, d, rep, a, base, settings, chash):
    name = run_name(a, doc, method, rep)
    words = target_words_for(a, d)
    cmd = build_cmd(method, d["source"], name, words, a)
    rec = {"doc": doc, "dataset": d["dataset"], "method": method, "rep": rep, "name": name,
           "target_words": words}
    if a.dry_run:
        print("  " + " ".join(cmd))
        return None

    wipe_outputs(name)
    usage_log = base / "usage" / f"{safe(doc)}_{method}_r{rep}.jsonl"
    usage_log.parent.mkdir(parents=True, exist_ok=True)
    if usage_log.exists():
        usage_log.unlink()
    env = dict(ENV)
    env["LLM_USAGE_LOG"] = str(usage_log)
    if settings.get("model"):
        env["MODEL_NAME"] = settings["model"]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    wall = time.time() - t0

    log = base / "logs" / f"{safe(doc)}_{method}_r{rep}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\n\n--- STDOUT ---\n{proc.stdout}\n"
                   f"--- STDERR ---\n{proc.stderr}", encoding="utf-8")

    rec["wall_seconds"] = round(wall, 1)
    rec["code_hash"] = chash
    if proc.returncode != 0:
        rec["status"] = "failed"
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown"]
        rec["error"] = tail[0][:300]
        return rec

    produced = expected_summary(method, name, words)
    meta = load_json(output_folder(method, name) / "metadata.json")
    if not produced.exists() or not meta:
        rec["status"] = "failed"
        rec["error"] = f"summary or metadata not found for {produced}"
        return rec
    if meta.get("resumed_from_chunk"):
        rec["status"] = "failed"
        rec["error"] = "run was resumed from saved progress - not comparable"
        return rec

    dest = base / "summaries" / safe(doc) / f"{method}_r{rep}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, dest)
    text = dest.read_text(encoding="utf-8")
    words_out = count_words(text)
    usage = meta.get("usage", {})
    flc = meta.get("final_length_control", {}) or {}

    rec.update({
        "status": "ok",
        "summary_path": str(dest),
        "words": words_out,
        "length_ok": abs(words_out - words) <= 0.10 * words,
        "document_tokens": meta.get("document_tokens"),
        "chunks": meta.get("chunks"),
        "model_calls": usage.get("api_requests"),
        "failed_requests": usage.get("failed_requests"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "llm_seconds": usage.get("latency_seconds_sum"),
        "runtime_seconds": meta.get("runtime_seconds"),
        "final_compression_calls": flc.get("compression_calls"),
        "final_truncated": flc.get("truncated"),
        "draft_words": flc.get("initial_words"),
        "fits_context": meta.get("fits_context"),
        "llm_settings": meta.get("llm_settings"),
    })
    return rec


def cmd_generate(a):
    docs = resolve_docs(a)
    methods = pick_methods(a)
    preflight(docs, need_refs=(a.target_length == "reference"))
    base = base_dir(a)
    runs_path = base / "runs.json"
    runs = load_json(runs_path) or {}
    settings = current_llm_settings(a)
    chash = code_hash()

    if not settings.get("model") and not a.dry_run:
        sys.exit("MODEL_NAME is not set (use .env or --model).")

    cfg = load_json(base / "config.json") or {}
    if runs and cfg and not a.force:
        problems = []
        if cfg.get("llm_settings") and cfg["llm_settings"] != settings:
            problems.append(f"LLM settings differ:\n    before: {cfg['llm_settings']}\n    now:    {settings}")
        if cfg.get("code_hash") and cfg["code_hash"] != chash:
            problems.append("prompts/code changed since the existing runs were generated")
        if problems:
            sys.exit("Refusing to mix incomparable runs in tag '" + a.tag + "':\n  - " +
                     "\n  - ".join(problems) +
                     "\nUse a new --tag, or --force to regenerate everything in this tag.")

    if not a.dry_run:
        save_json(base / "config.json", {
            **{k: v for k, v in vars(a).items() if k != "func"},
            "llm_settings": settings, "code_hash": chash})

    jobs = []
    for rep in range(1, a.reps + 1):
        for i, (doc, d) in enumerate(docs.items()):
            for method in method_order(methods, i + rep - 1):
                jobs.append((method, doc, d, rep))

    par = max(1, getattr(a, "parallel_jobs", 1))
    print(f"{len(jobs)} generation jobs ({'one at a time' if par == 1 else f'{par} in parallel'})  "
          f"model={settings.get('model')}  code={chash}\n")
    lock = threading.Lock()

    def run_job(n, method, doc, d, rep):
        key = f"{doc}|{method}|{rep}"
        done = runs.get(key)
        if (not a.force and done and done.get("status") == "ok"
                and Path(done.get("summary_path", "")).exists()):
            print(f"[{n}/{len(jobs)}] {key}: already done, skipping")
            return
        print(f"[{n}/{len(jobs)}] {key} ...", flush=True)
        rec = generate_one(method, doc, d, rep, a, base, settings, chash)
        if rec is None:
            return
        with lock:
            runs[key] = rec
            save_json(runs_path, runs)
        if rec["status"] == "ok":
            flag = "" if rec["length_ok"] else "  <-- LENGTH OUT OF +-10%"
            if rec.get("fits_context") is False:
                flag += "  <-- DOCUMENT TRUNCATED (direct)"
            print(f"    {key} ok: {rec['words']}/{rec['target_words']} words, {rec['model_calls']} calls, "
                  f"{rec['total_tokens']} tokens, {rec['runtime_seconds']}s{flag}", flush=True)
        else:
            print(f"    {key} FAILED: {rec['error']}  (see logs/{safe(doc)}_{method}_r{rep}.log)",
                  flush=True)

    if par == 1:
        for n, job in enumerate(jobs, 1):
            run_job(n, *job)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=par) as ex:
            for f in [ex.submit(run_job, n, *job) for n, job in enumerate(jobs, 1)]:
                f.result()

    if not a.dry_run:
        print(f"\nSaved run records: {runs_path}")
        print("Next: python compare_methods.py evaluate" + (f" --tag {a.tag}" if a.tag else ""))


# --------------------------------------------------------------------------
# EVALUATE
# --------------------------------------------------------------------------
def valid_standard(p: Path) -> bool:
    d = load_json(p)
    return bool(d) and "rouge_lsum_f1" in d


def valid_grounding(p: Path) -> bool:
    d = load_json(p)
    return (bool(d) and "document_level_verification" in d
            and not d.get("failed_chunks"))


def valid_judge(p: Path) -> bool:
    d = load_json(p)
    return bool(d) and isinstance(d.get("overall_score"), int)


def run_task(desc: str, cmd: list, out_path: Path, validator, base: Path):
    if validator(out_path):
        return desc, True, "cached"
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=ENV)
    if proc.returncode == 0 and validator(out_path):
        return desc, True, "done"
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown error"]
    with open(base / "errors.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {desc}\n"
                 f"  cmd: {' '.join(cmd)}\n  error: {tail[0]}\n\n")
    return desc, False, tail[0][:200]


def cmd_evaluate(a):
    docs = resolve_docs(a)
    methods = pick_methods(a)
    preflight(docs, need_refs=True)
    base = base_dir(a)
    runs = load_json(base / "runs.json") or {}
    if not runs:
        sys.exit("No runs.json found. Run the 'generate' step first.")

    py = sys.executable
    tasks = []
    for key, rec in runs.items():
        if rec.get("status") != "ok" or rec["doc"] not in docs or (
                rec["method"] not in methods and not (not a.methods and base_method(rec["method"]) in METHODS)):
            continue
        doc, method, rep = rec["doc"], rec["method"], rec["rep"]
        src, ref = docs[doc]["source"], docs[doc]["reference"]
        summary = rec["summary_path"]
        ed = base / "eval" / safe(doc) / f"{method}_r{rep}"
        std_cmd = [py, "human_reference_evaluation.py", str(ref), summary,
                   "--method", method, "--output-file", str(ed / "standard.json")]
        if a.no_semantic:
            std_cmd.append("--no-semantic")
        if a.bertscore:
            std_cmd.append("--bertscore")
        tasks.append((f"{key} standard", std_cmd, ed / "standard.json", valid_standard))
        if not a.skip_grounding:
            tasks.append((f"{key} grounding",
                          [py, "-m", "scripts.verify_summary_against_source", str(src), summary,
                           "--context-size", str(a.context_size),
                           "--output-file", str(ed / "grounding.json")],
                          ed / "grounding.json", valid_grounding))
        for i in range(1, a.judge_repeats + 1):
            out = ed / f"judge_{i}.json"
            tasks.append((f"{key} judge#{i}",
                          [py, "-m", "scripts.llm_judge_human_reference", str(ref), summary,
                           "--output-file", str(out), "--model", a.judge_model],
                          out, valid_judge))

    if a.dry_run:
        for desc, cmd, _, _ in tasks:
            print(f"{desc}: {' '.join(cmd)}")
        print(f"\n{len(tasks)} evaluation tasks")
        return

    cfg = load_json(base / "config.json") or {}
    cfg.update({"judge_model": a.judge_model, "judge_repeats": a.judge_repeats,
                "bertscore": a.bertscore})
    save_json(base / "config.json", cfg)

    print(f"{len(tasks)} evaluation tasks, {a.eval_workers} in parallel\n")
    failed = 0
    with ThreadPoolExecutor(max_workers=a.eval_workers) as pool:
        futures = [pool.submit(run_task, d, c, o, v, base) for d, c, o, v in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            desc, ok, msg = fut.result()
            if not ok:
                failed += 1
            print(f"[{n}/{len(tasks)}] {desc}: {'OK ' + msg if ok else 'FAILED ' + msg}")
    print(f"\nFinished. {failed} task(s) failed"
          + (f" (details in {base / 'errors.log'}); re-run 'evaluate' to retry them." if failed else "."))
    print("Next: python compare_methods.py report")


# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------
JUDGE_FIELDS = [
    ("coverage_score", "Coverage"),
    ("emphasis_score", "Emphasis"),
    ("reference_alignment_score", "Alignment"),
    ("relevance_score", "Relevance"),
    ("coherence_score", "Coherence"),
    ("overall_score", "Overall"),
]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def pm(m, s, nd=2):
    if m is None:
        return "-"
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} +-{s:.{nd}f}"


def length_bin(tokens):
    if tokens is None:
        return "unknown"
    for lo, hi, label in LENGTH_BINS:
        if lo <= tokens < hi:
            return label
    return "unknown"


STD_KEYS = ["rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "rouge_lsum_f1", "rouge_1_recall",
            "rouge_lsum_recall", "semantic_similarity", "bertscore_f1"]
EFF_KEYS = ["calls", "failed_requests", "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "total_tokens", "llm_seconds", "runtime"]


def cell_metrics(base: Path, doc: str, method: str, recs: list) -> dict:
    ok = [r for r in recs if r.get("status") == "ok"]
    c = {"n_reps": len(ok), "failed_runs": len(recs) - len(ok)}
    c["dataset"] = (recs[0].get("dataset") if recs else None)
    c["document_tokens"] = median([r.get("document_tokens") for r in ok])
    c["target_words"] = median([r.get("target_words") for r in ok])
    c["words"] = mean([r.get("words") for r in ok])
    c["length_ok"] = all(r.get("length_ok") for r in ok) if ok else None
    c["calls"] = median([r.get("model_calls") for r in ok])
    c["failed_requests"] = median([r.get("failed_requests") for r in ok])
    for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "llm_seconds"):
        c[k] = median([r.get(k) for r in ok])
    c["runtime"] = median([r.get("runtime_seconds") for r in ok])
    c["truncated_input"] = any(r.get("fits_context") is False for r in ok)
    c["final_truncated"] = sum(1 for r in ok if r.get("final_truncated"))

    std, ground, bad, judge = [], [], 0, {k: [] for k, _ in JUDGE_FIELDS}
    incomplete_ground = 0
    for r in ok:
        ed = base / "eval" / safe(doc) / f"{method}_r{r['rep']}"
        s = load_json(ed / "standard.json")
        if s:
            std.append(s)
        g = load_json(ed / "grounding.json")
        if g and "document_level_verification" in g:
            if g.get("failed_chunks"):
                incomplete_ground += 1
            v = g["document_level_verification"]
            ground.append(v.get("support_percentage"))
            bad += (v.get("partial", 0) + v.get("unsupported", 0) + v.get("contradicted", 0))
        for p in sorted(ed.glob("judge_*.json")):
            if p.name.endswith("_raw_debug.json"):
                continue
            j = load_json(p)
            if j and isinstance(j.get("overall_score"), int):
                for k, _ in JUDGE_FIELDS:
                    judge[k].append(j.get(k))

    c["len_ratio"] = mean([s.get("length_ratio") for s in std])
    for k in STD_KEYS:
        c[k] = mean([s.get(k) for s in std])
    c["support_pct"] = mean(ground)
    c["not_supported_stmts"] = bad if ground else None
    c["grounding_incomplete"] = incomplete_ground
    for k, _ in JUDGE_FIELDS:
        c[f"judge_{k}"] = mean(judge[k])
    c["judge_overall_sd"] = sd(judge["overall_score"])
    c["judge_n"] = len(judge["overall_score"])
    return c


def macro(cells: list) -> dict:
    keys = ["words", "len_ratio", "support_pct", "judge_overall_sd"] + STD_KEYS + EFF_KEYS + \
        [f"judge_{k}" for k, _ in JUDGE_FIELDS]
    out = {k: mean([c.get(k) for c in cells]) for k in keys}
    for k in ("calls", "total_tokens", "llm_seconds", "runtime"):
        out[f"{k}_total"] = sum(c[k] for c in cells if c.get(k) is not None)
    out["docs"] = len([c for c in cells if c["n_reps"]])
    # documents that actually have evaluation results (judge or metrics)
    out["docs_evaluated"] = len([c for c in cells if c.get("judge_n") or c.get("rouge_1_f1") is not None])
    return out


def overall_table(L, methods, cell_groups, title):
    L.append(f"## {title}\n")
    L.append("| Method | Docs evaluated | Judge overall | Judge sd | Coverage | Emphasis | Alignment | "
             "R-1 | R-2 | R-Lsum | R-Lsum recall | Semantic | BERTScore | Grounding % |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        mc = macro(cell_groups[m])
        L.append(f"| {label(m)} | {mc['docs_evaluated']}/{mc['docs']} | {f(mc['judge_overall_score'])} | "
                 f"{f(mc['judge_overall_sd'])} | {f(mc['judge_coverage_score'])} | "
                 f"{f(mc['judge_emphasis_score'])} | {f(mc['judge_reference_alignment_score'])} | "
                 f"{f(mc['rouge_1_f1'], 3)} | {f(mc['rouge_2_f1'], 3)} | {f(mc['rouge_lsum_f1'], 3)} | "
                 f"{f(mc['rouge_lsum_recall'], 3)} | {f(mc['semantic_similarity'], 3)} | "
                 f"{f(mc['bertscore_f1'], 3)} | {f(mc['support_pct'], 1)} |")
    L.append("")


def cmd_report(a):
    base = base_dir(a)
    runs = load_json(base / "runs.json")
    if not runs:
        sys.exit("No runs.json found. Run 'generate' (and 'evaluate') first.")
    cfg = load_json(base / "config.json") or {}

    docs = []
    for r in runs.values():
        if r["doc"] not in docs:
            docs.append(r["doc"])
    present = []
    for r in runs.values():
        if r["method"] not in present:
            present.append(r["method"])
    methods = [m for b in METHODS for m in sorted(present) if base_method(m) == b]
    # --docs / --methods restrict the report (e.g. to what has been evaluated so far)
    if a.docs:
        wanted = [d.strip() for d in a.docs.split(",") if d.strip()]
        docs = [d for d in docs if d in wanted]
    if a.methods:
        wanted = [m.strip() for m in a.methods.split(",") if m.strip()]
        methods = [m for m in methods if m in wanted]
    if not docs or not methods:
        sys.exit("Nothing to report for the selected --docs / --methods.")

    table = {}
    for d in docs:
        for m in methods:
            recs = [r for r in runs.values() if r["doc"] == d and r["method"] == m]
            table[(d, m)] = cell_metrics(base, d, m, recs)

    # only documents where EVERY method succeeded enter the averages, so no
    # method is averaged over an easier subset of documents
    complete_docs = [d for d in docs if all(table[(d, m)]["n_reps"] for m in methods)]
    dropped = [d for d in docs if d not in complete_docs]

    report_dir = base / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV ----
    csv_path = report_dir / "results.csv"
    cols = ["doc", "dataset", "method", "n_reps", "document_tokens", "target_words", "words",
            "length_ok", "len_ratio"] + STD_KEYS + ["support_pct", "not_supported_stmts"] + \
           [f"judge_{k}" for k, _ in JUDGE_FIELDS] + ["judge_overall_sd", "judge_n"] + EFF_KEYS + \
           ["truncated_input", "final_truncated"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in docs:
            for m in methods:
                c = table[(d, m)]
                w.writerow([d, c.get("dataset"), m] + [c.get(k) for k in cols[3:]])
        for m in methods:
            mc = macro([table[(d, m)] for d in complete_docs])
            w.writerow(["MACRO_AVG", "", m] + [mc.get(k, "") for k in cols[3:]])

    # ---- Markdown ----
    L = []
    L.append(f"# Method comparison: {a.tag}\n")
    s = cfg.get("llm_settings") or {}
    L.append(f"Model: {s.get('model')} (temperature={s.get('temperature')}, seed={s.get('seed')}, "
             f"reasoning={s.get('reasoning_effort')}, provider={s.get('provider')}). "
             f"Target length: {cfg.get('target_length')} "
             f"({cfg.get('summary_words') if cfg.get('target_length') == 'fixed' else 'x' + str(cfg.get('reference_ratio')) + ' reference'}), "
             f"context_size={cfg.get('context_size')}, workers={cfg.get('workers')}, reps={cfg.get('reps')}, "
             f"judge={cfg.get('judge_model')} x{cfg.get('judge_repeats')}, code={cfg.get('code_hash')}.\n")
    L.append("Averages use only the documents on which every method produced a summary "
             f"({len(complete_docs)}/{len(docs)}). Judge scores are 1-5; 'Judge sd' is the spread "
             "between repeated judge runs on the SAME summary - smaller differences are noise.\n")
    if dropped:
        L.append(f"Excluded from averages (some method failed): {', '.join(dropped)}\n")

    overall_table(L, methods, {m: [table[(d, m)] for d in complete_docs] for m in methods},
                  "Overall (macro-average over documents)")

    datasets = sorted({table[(d, methods[0])].get("dataset") or "-" for d in complete_docs})
    if len(datasets) > 1:
        for ds in datasets:
            ds_docs = [d for d in complete_docs if (table[(d, methods[0])].get("dataset") or "-") == ds]
            overall_table(L, methods, {m: [table[(d, m)] for d in ds_docs] for m in methods},
                          f"Dataset: {ds} ({len(ds_docs)} docs)")

    L.append("## Efficiency (identical accounting for every method)\n")
    L.append("Calls = every API request incl. retries. 'LLM s' = summed latency of all requests "
             "(independent of parallelism; the fair compute-time measure). 'Wall s' depends on "
             "parallelism (Sequential and Direct cannot be parallelised) and on API load. "
             "Values are per-document medians, averaged over documents.\n")
    L.append("| Method | Calls | Failed req. | Prompt tok | Completion tok | Reasoning tok | "
             "Total tok | LLM s | Wall s |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        mc = macro([table[(d, m)] for d in complete_docs])
        L.append(f"| {label(m)} | {f(mc['calls'], 1)} | {f(mc['failed_requests'], 1)} | "
                 f"{f(mc['prompt_tokens'], 0)} | {f(mc['completion_tokens'], 0)} | "
                 f"{f(mc['reasoning_tokens'], 0)} | {f(mc['total_tokens'], 0)} | "
                 f"{f(mc['llm_seconds'], 1)} | {f(mc['runtime'], 1)} |")
    L.append("")

    L.append("## By document length (document tokens)\n")
    bins = [b for _, _, b in LENGTH_BINS]
    present = [b for b in bins if any(length_bin(table[(d, methods[0])]["document_tokens"]) == b
                                      for d in complete_docs)]
    if present:
        L.append("Judge overall / ROUGE-Lsum F1 per length bin.\n")
        L.append("| Method | " + " | ".join(
            f"{b} (n={sum(1 for d in complete_docs if length_bin(table[(d, methods[0])]['document_tokens']) == b)})"
            for b in present) + " |")
        L.append("|---|" + "---:|" * len(present))
        for m in methods:
            cells = []
            for b in present:
                group = [table[(d, m)] for d in complete_docs
                         if length_bin(table[(d, methods[0])]["document_tokens"]) == b]
                mc = macro(group)
                cells.append(f"{f(mc['judge_overall_score'])} / {f(mc['rouge_lsum_f1'], 3)}")
            L.append(f"| {label(m)} | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## Per document\n")
    for d in docs:
        c0 = table[(d, methods[0])]
        L.append(f"### {d}  ({c0.get('dataset')}, {f(c0.get('document_tokens'), 0)} tokens, "
                 f"target {f(c0.get('target_words'), 0)} words)\n")
        L.append("| Method | Words | R-1 | R-2 | R-Lsum | Semantic | Grounding % | Not-supp. | "
                 "Judge overall (mean +-sd) | Cov | Emph | Align | Calls | Tokens |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in methods:
            c = table[(d, m)]
            L.append(f"| {label(m)} | {f(c['words'], 0)} | {f(c['rouge_1_f1'], 3)} | "
                     f"{f(c['rouge_2_f1'], 3)} | {f(c['rouge_lsum_f1'], 3)} | "
                     f"{f(c['semantic_similarity'], 3)} | {f(c['support_pct'], 1)} | "
                     f"{'-' if c['not_supported_stmts'] is None else c['not_supported_stmts']} | "
                     f"{pm(c['judge_overall_score'], c['judge_overall_sd'])} | "
                     f"{f(c['judge_coverage_score'], 1)} | {f(c['judge_emphasis_score'], 1)} | "
                     f"{f(c['judge_reference_alignment_score'], 1)} | {f(c['calls'], 0)} | "
                     f"{f(c['total_tokens'], 0)} |")
        L.append("")

    flags = []
    settings_seen = {json.dumps(r.get("llm_settings"), sort_keys=True)
                     for r in runs.values() if r.get("status") == "ok"}
    if len(settings_seen) > 1:
        flags.append(f"RUNS USE DIFFERENT LLM SETTINGS ({len(settings_seen)} variants) - "
                     "the comparison is not controlled")
    hashes = {r.get("code_hash") for r in runs.values() if r.get("status") == "ok"}
    if len(hashes) > 1:
        flags.append(f"RUNS WERE GENERATED WITH DIFFERENT PROMPTS/CODE ({sorted(map(str, hashes))})")
    for (d, m), c in table.items():
        if c["failed_runs"]:
            flags.append(f"{d}/{m}: {c['failed_runs']} failed generation run(s) (see logs/)")
        if c["n_reps"] == 0:
            flags.append(f"{d}/{m}: no successful summary")
            continue
        if c["truncated_input"]:
            flags.append(f"{d}/{m}: document did not fit the context and was truncated")
        if c["length_ok"] is False:
            flags.append(f"{d}/{m}: length outside +-10% of target "
                         f"(avg {f(c['words'], 0)} vs {f(c['target_words'], 0)} words)")
        if c["grounding_incomplete"]:
            flags.append(f"{d}/{m}: grounding verification incomplete (some chunks failed)")
        if c["judge_n"] == 0:
            flags.append(f"{d}/{m}: no judge results")
        elif c["judge_n"] < (cfg.get("judge_repeats") or 1) * c["n_reps"]:
            flags.append(f"{d}/{m}: only {c['judge_n']} judge result(s)")
    L.append("## Flags\n")
    L.append("\n".join(f"- {x}" for x in flags) if flags else "None.")
    L.append("")

    md_path = report_dir / "results.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\nSaved: {md_path}\n       {csv_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Fair comparison of long-document summarization strategies")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--tag", default=None, help="output folder name")
        sp.add_argument("--summary-words", type=int, default=1000)
        sp.add_argument("--target-length", choices=["fixed", "reference"], default="fixed",
                        help="fixed = --summary-words for every document; "
                             "reference = length of each document's human reference")
        sp.add_argument("--reference-ratio", type=float, default=1.0)
        sp.add_argument("--context-size", type=int, default=8000)
        sp.add_argument("--docs", default="", help="comma-separated subset of documents")
        sp.add_argument("--doc", action="append", help="add/override a document: name=path")
        sp.add_argument("--manifest", default=None, help="JSONL with name/source/reference/dataset")
        sp.add_argument("--methods", default="", help="comma-separated subset of methods")
        sp.add_argument("--dry-run", action="store_true")

    g = sub.add_parser("generate", help="generate summaries with every method")
    common(g)
    g.add_argument("--model", default=None, help="summarization model (default: MODEL_NAME)")
    g.add_argument("--reps", type=int, default=1)
    g.add_argument("--workers", type=int, default=5)
    g.add_argument("--max-claims", type=int, default=10)
    g.add_argument("--max-sentences", type=int, default=10)
    g.add_argument("--chunk-summary-words", type=int, default=500)
    g.add_argument("--intermediate-summary-words", type=int, default=800)
    g.add_argument("--merge-group-size", type=int, default=4)
    g.add_argument("--max-input-tokens", type=int, default=100000,
                   help="largest single prompt (Direct input, Map-Reduce reduce)")
    g.add_argument("--direct-overflow", choices=["truncate", "fail"], default="truncate")
    g.add_argument("--force", action="store_true")
    g.add_argument("--parallel-jobs", type=int, default=1,
                   help="run N generation jobs at once (faster; wall time is then not comparable "
                        "- use for prompt experiments, keep 1 for the final comparison)")
    g.add_argument("--reuse-support", default=None, metavar="TAG:METHOD",
                   help="prompt experiments: reuse the support sentences of TAG/METHOD (same doc "
                        "and rep) so only the final summary step runs, e.g. "
                        "exp1_overview:key_sentences_overview")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="standard metrics, grounding and repeated LLM judge")
    common(e)
    e.add_argument("--judge-model", default="deepseek/deepseek-v3.2")
    e.add_argument("--judge-repeats", type=int, default=3)
    e.add_argument("--skip-grounding", action="store_true")
    e.add_argument("--no-semantic", action="store_true")
    e.add_argument("--bertscore", action="store_true")
    e.add_argument("--eval-workers", type=int, default=3)
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="aggregate everything into tables and a CSV")
    common(r)
    r.set_defaults(func=cmd_report)

    a = p.parse_args()
    if a.tag is None:
        a.tag = (f"main_{a.summary_words}" if a.target_length == "fixed"
                 else f"main_ref{a.reference_ratio:g}")
    a.func(a)


if __name__ == "__main__":
    main()