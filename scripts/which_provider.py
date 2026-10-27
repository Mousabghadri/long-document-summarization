
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import _get_client  # noqa: E402  (loads .env)

p = argparse.ArgumentParser()
p.add_argument("--model", default="deepseek/deepseek-v3.2")
p.add_argument("--n", type=int, default=6)
p.add_argument("--pin", default="", help="provider name to pin (no fallbacks)")
a = p.parse_args()

counts = collections.Counter()
for i in range(a.n):
    args = {"model": a.model, "max_tokens": 5,
            "messages": [{"role": "user", "content": "Reply with the word OK."}]}
    if a.pin:
        args["extra_body"] = {"provider": {"order": [a.pin], "allow_fallbacks": False}}
    try:
        r = _get_client().chat.completions.create(**args)
        prov = getattr(r, "provider", None) or (r.model_extra or {}).get("provider", "?")
    except Exception as e:  # noqa: BLE001
        prov = f"ERROR: {str(e)[:120]}"
    counts[prov] += 1
    print(f"request {i + 1}: {prov}")

print("\nSummary:", dict(counts))