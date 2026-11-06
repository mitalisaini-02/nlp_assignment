"""Parse the `final_answer_sources` column of the training CSV.

The column is a stringified list of dict-strings, newline separated (no commas), e.g.
    ["{'source': 'https://workdrive.zoho.in/file/<id>', 'page': '1,2', 'sourceName': '...'}"
     "{'source': ..., 'page': None, ...}"]
Some entries are junk (free-text notes, zoho sheets, leading spaces, empty strings).
"""
import ast
import re

BLOCK_RE = re.compile(r"\{[^{}]*\}")
FIELD_RE = {
    k: re.compile(r"'%s':\s*(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|(None|\d+))" % k)
    for k in ("source", "page", "sourceName", "sourceType")
}
FILE_ID_RE = re.compile(r"/file/([A-Za-z0-9]+)")


def _fields_from_block(block):
    try:
        d = ast.literal_eval(block)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    d = {}
    for k, rx in FIELD_RE.items():
        m = rx.search(block)
        if m:
            v = m.group(1) if m.group(1) is not None else (m.group(2) if m.group(2) is not None else m.group(3))
            d[k] = None if v == "None" else v
    return d


def parse_pages(page):
    """'45, 46' -> [45, 46]; '3-5' -> [3, 4, 5]; None -> []."""
    if page is None:
        return []
    s = str(page)
    pages = []
    for part in re.split(r"[,;&]|\band\b", s):
        part = part.strip()
        m = re.fullmatch(r"(\d+)\s*[-\u2013]\s*(\d+)", part)
        if m and int(m.group(2)) - int(m.group(1)) < 200:
            pages.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.fullmatch(r"\d+", part):
            pages.append(int(part))
    return sorted(set(pages))


def classify(url):
    url = (url or "").strip()
    if not url.lower().startswith("http"):
        return "invalid", None
    if re.match(r"https?://workdrive\.zoho(external)?\.in/", url):
        m = FILE_ID_RE.search(url)
        return ("zoho", m.group(1)) if m else ("invalid", None)
    if "sheet.zoho" in url or "docs.zoho" in url:
        return "zoho_other", None
    return "external", None


def parse_sources(cell):
    """Return a list of citation dicts: kind, file_id, url, pages, page_raw, name, source_type."""
    out = []
    for block in BLOCK_RE.findall(cell or ""):
        d = _fields_from_block(block)
        raw = (d.get("source") or "").strip()
        # some entries concatenate several URLs in one field
        urls = [u.strip() for u in re.split(r"(?=https?://)", raw) if u.strip()] or [raw]
        for url in urls:
            kind, fid = classify(url)
            out.append({
                "kind": kind,
                "file_id": fid,
                "url": url,
                "pages": parse_pages(d.get("page")),
                "page_raw": None if d.get("page") is None else str(d.get("page")),
                "name": (d.get("sourceName") or "").strip(),
                "source_type": d.get("sourceType"),
            })
    return out
