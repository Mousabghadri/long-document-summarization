
import argparse
import json
import sys
import time
import urllib.request

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import count_tokens  # noqa: E402
from src.document_loader import html_to_markdown  # noqa: E402


DEV_ACTS = {  # name -> CELEX number
    "ai_act": "32024R1689",
    "gdpr": "32016R0679",
    "dga": "32022R0868",
    "data_act": "32023R2854",
}

URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"


p = argparse.ArgumentParser()

p.add_argument(
    "--celex",
    default=None
)

p.add_argument(
    "--name",
    default=None
)

a = p.parse_args()


acts = (
    {a.name or a.celex: a.celex}
    if a.celex
    else DEV_ACTS
)

out_dir = ROOT / "data" / "html"

out_dir.mkdir(
    parents=True,
    exist_ok=True
)


CELLAR = "http://publications.europa.eu/resource/celex/{celex}"


def download(celex):
    """EUR-Lex first; it may answer bots with an empty page, then try the
    Publications Office (Cellar) endpoint with content negotiation."""

    tries = [
        (
            URL.format(celex=celex),
            {
                "User-Agent": "Mozilla/5.0"
            }
        ),
        (
            CELLAR.format(celex=celex),
            {
                "Accept": "application/xhtml+xml, text/html",
                "Accept-Language": "eng",
                "User-Agent": "Mozilla/5.0"
            }
        ),
    ]

    for url, headers in tries:

        try:

            with urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    headers=headers
                ),
                timeout=120
            ) as r:

                html = r.read().decode(
                    "utf-8",
                    errors="replace"
                )

            if len(html) > 20000 and "Article" in html:
                return html

        except Exception as e:  # noqa: BLE001

            print(
                f"  {url}: {str(e)[:100]}"
            )

    return ""


failed = []


for name, celex in acts.items():

    path = out_dir / f"{name}.html"

    html = (
        path.read_text(
            encoding="utf-8",
            errors="replace"
        )
        if path.exists()
        else ""
    )

    if len(html) > 20000:

        print(
            f"{name:10s} using existing {path.name}"
        )

    else:

        html = download(celex)

        if not html:

            failed.append((name, celex))

            print(
                f"{name:10s} CELEX {celex}: download failed"
            )

            continue

        path.write_text(
            html,
            encoding="utf-8"
        )

        time.sleep(1)

    md = html_to_markdown(html)

    (out_dir / f"{name}.md").write_text(
        md,
        encoding="utf-8"
    )

    print(
        f"{name:10s} CELEX {celex}: "
        f"{len(html):,} bytes HTML -> "
        f"{count_tokens(md):,} tokens Markdown"
    )


if failed:

    print(
        "\nEUR-Lex blocked the automatic download. "
        "Open these pages in your browser, save each one"
    )

    print(
        f"(Ctrl+S, 'Webpage, HTML only') into {out_dir} "
        "with the name shown, then run this script again:"
    )

    for name, celex in failed:

        print(
            f"  {name}.html  <-  {URL.format(celex=celex)}"
        )

    sys.exit(1)


if not a.celex:

    clean = ROOT / "data" / "manifest_clean.jsonl"

    refs = {}

    if clean.exists():

        for line in clean.read_text(
            encoding="utf-8"
        ).splitlines():

            if line.strip():

                e = json.loads(line)

                refs[e["name"]] = e

    rows = []

    for name in DEV_ACTS:

        e = refs.get(
            name,
            {
                "name": name,
                "reference": f"data/references_clean/{name}_human_overview.txt",
                "dataset": "eu_regulations"
            }
        )

        rows.append({
            **e,
            "source": f"data/html/{name}.md"
        })

    man = ROOT / "data" / "manifest_html.jsonl"

    man.write_text(
        "\n".join(
            json.dumps(r)
            for r in rows
        ) + "\n",
        encoding="utf-8"
    )

    print("Manifest:", man)