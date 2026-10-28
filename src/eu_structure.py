
import re

_OPERATIVE = re.compile(
    r"(?:HA(?:VE|S)\s+ADOPTED\s+TH(?:IS|E\s+PRESENT)\s+(?:REGULATION|DIRECTIVE|DECISION|RECOMMENDATION)"
    r"|HAVE\s+AGREED\s+AS\s+FOLLOWS)[^\n]*", re.IGNORECASE)
_PREAMBLE_START = re.compile(
    r"^\s*(?:THE\s+EUROPEAN\s+PARLIAMENT|THE\s+COUNCIL|THE\s+EUROPEAN\s+COMMISSION|"
    r"Having\s+regard|Whereas)", re.IGNORECASE | re.MULTILINE)


def split_act(text: str) -> dict:
    """{'title', 'preamble', 'operative', 'found'}; found=False -> no split."""
    m = _OPERATIVE.search(text)
    if not m:
        return {"title": "", "preamble": "", "operative": text, "found": False}
    head, operative = text[:m.start()], text[m.end():]
    p = _PREAMBLE_START.search(head)
    title = head[:p.start()] if p else head[:1500]
    preamble = head[p.start():] if p else head[1500:]
    return {"title": title.strip(), "preamble": preamble.strip(),
            "operative": operative.strip(), "found": True}


def title_of(text: str, max_words: int = 80) -> str:
    t = split_act(text)["title"] or text
    return " ".join(t.split()[:max_words])
