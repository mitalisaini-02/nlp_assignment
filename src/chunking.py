"""Phase 1: turn extracted page text into retrieval chunks.

Per document: drop pages that are not usable text, strip repeated headers/footers, undo line-break hyphenation,
then cut each page into overlapping word windows. A chunk never spans two pages, so every chunk keeps an exact
(doc_id, page) citation. Source is pure ASCII on purpose (invisible characters get mangled by copy-paste).
"""
import re
from collections import Counter

USABLE_PAGE_QUALITY = {
    "good": {"ok", "numeric", "table"},
    "indic_unicode": {"indic_unicode", "ok", "numeric", "table"},
    "mixed": {"ok", "table", "indic_unicode"},
}
CHUNK_WORDS = 250     # roughly 330 tokens
OVERLAP_WORDS = 50
MIN_PAGE_WORDS = 15   # shorter pages are dropped (page numbers, captions)
MIN_TAIL_WORDS = 40   # a shorter final window is merged into the previous chunk


def _norm_line(line):
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def boilerplate_lines(pages):
    """Short lines that repeat on many pages of one document (running headers, footers, page numbers)."""
    n = len(pages)
    if n < 4:
        return set()
    seen = Counter()
    for p in pages:
        seen.update({_norm_line(l) for l in p["text"].split("\n") if 0 < len(l.strip()) <= 100})
    limit = max(3, int(0.4 * n))
    return {l for l, c in seen.items() if c >= limit and l}


def clean_text(text, drop=frozenset()):
    lines = [l for l in text.split("\n") if _norm_line(l) not in drop]
    t = "\n".join(lines)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)  # re-join words split across lines
    return re.sub(r"\s+", " ", t).strip()


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS, min_tail=MIN_TAIL_WORDS):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    chunks, i = [], 0
    while i < len(words):
        w = words[i:i + size]
        if chunks and len(w) < min_tail:
            tail = words[i + overlap:]  # words not already present in the previous chunk
            if tail:
                chunks[-1] = chunks[-1] + " " + " ".join(tail)
            break
        chunks.append(" ".join(w))
        if i + size >= len(words):
            break
        i += step
    return chunks


def process_doc(pages, verdict):
    """pages: list of page dicts of ONE document. Returns list of chunk dicts (empty if the doc is not usable)."""
    ok_q = USABLE_PAGE_QUALITY.get(verdict)
    if not ok_q:
        return []
    drop = boilerplate_lines(pages)
    out = []
    for p in sorted(pages, key=lambda p: p["page"]):
        if p["quality"] not in ok_q:
            continue
        t = clean_text(p["text"], drop)
        if len(t.split()) < MIN_PAGE_WORDS:
            continue
        for k, c in enumerate(chunk_words(t)):
            out.append({"doc_id": p["doc_id"], "page": p["page"], "chunk": k, "script": p["script"],
                        "quality": p["quality"], "n_words": len(c.split()), "text": c})
    return out
