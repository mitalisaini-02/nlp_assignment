"""Download source PDFs, extract per-page text, score extraction quality.

Designed for Kaggle (20 GB working-dir cap): PDFs are downloaded to a temp dir, text is extracted,
and the PDF is deleted unless keep_pdfs=True. Only compressed text shards are kept.

Outputs (out_dir):
  pages_XXXXX.jsonl.gz   one JSON per page: doc_id, page (1-based), n_chars, script, quality, sw, text
  status.jsonl           one JSON per processed doc (append-only; used for resume)
  doc_summary.csv        written by summarize(): one row per doc with quality verdict
"""
import csv
import gzip
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from urllib.parse import unquote

import requests

ZOHO_CDN = "https://files-accl.zohoexternal.in/public/workdrive-external/download/{}?x-cli-msg=null"
UA = {"User-Agent": "Mozilla/5.0 (compatible; academic-nlp-assignment)"}
MAX_BYTES = 400 * 1024 * 1024
DOC_DEADLINE_S = 240
EXTRACT_DOC_S = 120  # hard wall-clock limit per PDF; a hung extraction is killed and the doc marked extract_timeout
TERMINAL = {"ok", "not_pdf", "http_404", "http_403", "too_large", "extract_error", "extract_timeout"}

SCRIPTS = {
    "devanagari": (0x0900, 0x097F), "bengali": (0x0980, 0x09FF), "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF), "odia": (0x0B00, 0x0B7F), "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F), "kannada": (0x0C80, 0x0CFF), "malayalam": (0x0D00, 0x0D7F),
}
STOP = set("""the of and in to a is for with on are be by as at or from that this it an can should per
which also not may more than will have has been used use when if into these their such other all
during after before between each only one two high low early late crop plant plants""".split())
WORD_RE = re.compile(r"[A-Za-z]+")
MIDCAP_RE = re.compile(r"[a-z][A-Z]")
ODDCHAR_RE = re.compile(r"[;{}\[\]<>\\|~^`@]")
ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
EDGE_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")
LEGACY_THR = 0.25  # page-level legacy_score at or above this => garbled (legacy-font) page


# ----------------------------------------------------------------------------- download
def _safe_name(cd):
    if not cd:
        return ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    return unquote(m.group(1)) if m else ""


def download(doc, dest_dir, session, retries=4):
    """Returns (status, path, info). status in {ok, not_pdf, http_NNN, timeout, too_large}."""
    url = ZOHO_CDN.format(doc["doc_id"]) if doc["kind"] == "zoho" else doc["url"]
    path = os.path.join(dest_dir, doc["doc_id"] + ".pdf")
    last = "timeout"
    deadline = time.time() + DOC_DEADLINE_S  # hard wall-clock cap across all attempts
    for attempt in range(retries):
        if time.time() > deadline:
            break
        try:
            with session.get(url, headers=UA, stream=True, timeout=(10, 30), allow_redirects=True) as r:
                if r.status_code != 200:
                    last = "http_%d" % r.status_code
                    if r.status_code in (403, 404):
                        return last, None, {}
                    raise RuntimeError(last)
                size = 0
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            return "too_large", None, {}
                        if time.time() > deadline:  # trickling connection
                            raise TimeoutError("deadline")
                        f.write(chunk)
                info = {"bytes": size, "filename": _safe_name(r.headers.get("Content-Disposition")),
                        "ctype": r.headers.get("Content-Type", "")}
            with open(path, "rb") as f:
                head = f.read(1024)
            if b"%PDF-" not in head:
                os.remove(path)
                return "not_pdf", None, info
            return "ok", path, info
        except Exception as e:  # network error, timeout, 5xx/429
            last = last if last.startswith("http_") else "timeout"
            time.sleep(min(30, 2 ** attempt) + random.random())
    return last, None, {}


# ----------------------------------------------------------------------------- extraction
def page_stats(text):
    t = text.strip()
    n = len(t)
    if n < 40:
        return {"n_chars": n, "script": "none", "quality": "empty", "sw": 0.0}
    letters = [c for c in t if c.isalpha()]
    counts = Counter()
    for c in letters:
        o = ord(c)
        if o < 0x0900:  # all Indic blocks start at 0x0900, so skip the range scan for Latin-range letters
            counts["latin" if o < 0x250 else "other"] += 1
            continue
        for name, (lo, hi) in SCRIPTS.items():
            if lo <= o <= hi:
                counts[name] += 1
                break
        else:
            counts["latin" if o < 0x250 else "other"] += 1
    nl = max(1, len(letters))
    top, topn = counts.most_common(1)[0] if counts else ("none", 0)
    repl_r = t.count(chr(0xFFFD)) / n  # real decoding failures
    toks = [w.lower() for w in WORD_RE.findall(t)]
    sw = sum(1 for w in toks if w in STOP) / max(1, len(toks))
    nonspace = max(1, sum(1 for c in t if not c.isspace()))
    digit_r = sum(c.isdigit() for c in t) / nonspace
    ls = legacy_score(t)
    if top not in ("latin", "other") and topn / nl >= 0.4:
        q = "indic_unicode"
    elif repl_r > 0.02 or (ls is not None and ls >= LEGACY_THR):
        q = "garbled"          # legacy-font Hindi/Marathi/Telugu rendered as Latin gibberish
    elif sw >= 0.12 and len(toks) >= 15:
        q = "ok"
    elif digit_r > 0.35:
        q = "numeric"
    elif len(toks) < 15:
        q = "sparse"
    else:
        q = "table"            # clean Latin text with few stopwords: dosage tables, lists, web printouts
    return {"n_chars": n, "script": top, "quality": q, "sw": round(sw, 3)}


def legacy_score(text):
    """Share of tokens that look like legacy-font gibberish: a capital inside a word (xUus, egRoiw.kZ) or symbols such as
    ; { } [ ] < > | inside a word. English prose and English tables score ~0, Kruti-Dev-style text scores 0.4-1.0.
    Returns None for pages with fewer than 10 candidate tokens."""
    toks = [x for x in text.split() if len(x) >= 3 and any(c.isalpha() for c in x)]
    if len(toks) < 10:
        return None
    # chemical formulas (NaOH, CaCl2, ZnSO4, NH4Cl) have mid-word capitals too, but are element-symbol sequences
    mid = sum(1 for x in toks if MIDCAP_RE.search(x) and not FORMULA_RE.match(EDGE_RE.sub("", x)))
    odd = sum(1 for x in toks if ODDCHAR_RE.search(x) and ASCII_ALPHA_RE.search(x))
    return (mid + odd) / len(toks)


def extract(path):
    """Worker: returns (doc_id, pages|None, err)."""
    import pymupdf as fitz  # `import fitz` is the deprecated alias and warns in every worker
    doc_id = os.path.basename(path)[:-4]
    try:
        d = fitz.open(path)
        if d.needs_pass and not d.authenticate(""):
            return doc_id, None, "encrypted"
        pages = []
        for i in range(d.page_count):
            try:
                text = d[i].get_text("text")
            except Exception:
                text = ""
            st = page_stats(text)
            st.update(doc_id=doc_id, page=i + 1, text=text)
            pages.append(st)
        d.close()
        return doc_id, pages, None
    except Exception as e:
        return doc_id, None, "%s: %s" % (type(e).__name__, str(e)[:120])


def _extract_worker(path, out_path):
    """Child process: extract one PDF and write the result to disk (so the parent can kill us safely)."""
    doc_id, pages, err = extract(path)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump([pages, err], f, ensure_ascii=False)


def extract_all(paths, procs, per_doc_s=None):
    """Extract many PDFs, at most `procs` at a time, each in its own process with a hard wall-clock limit.
    A PDF that hangs (corrupt file, pathological page) is killed after per_doc_s and reported as 'timeout'
    without affecting the others. Returns {doc_id: (pages|None, err|None)}."""
    import multiprocessing as mp
    per_doc_s = per_doc_s or EXTRACT_DOC_S
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    results, pending, running = {}, list(paths), {}
    while pending or running:
        while pending and len(running) < procs:
            p = pending.pop(0)
            out = p[:-4] + ".json.gz"
            pr = ctx.Process(target=_extract_worker, args=(p, out))
            pr.start()
            running[pr] = (p, out, time.time())
        time.sleep(0.2)
        for pr, (p, out, t0) in list(running.items()):
            doc_id = os.path.basename(p)[:-4]
            if not pr.is_alive():
                pr.join()
                if os.path.exists(out):
                    try:
                        pages, err = json.load(gzip.open(out, "rt", encoding="utf-8"))
                        results[doc_id] = (pages, err)
                    except Exception as e:
                        results[doc_id] = (None, "bad result file: %s" % e)
                    os.remove(out)
                else:  # e.g. killed by the OOM killer
                    results[doc_id] = (None, "worker died (exit code %s)" % pr.exitcode)
                del running[pr]
            elif time.time() - t0 > per_doc_s:
                pr.kill()
                pr.join()
                results[doc_id] = (None, "timeout")
                if os.path.exists(out):
                    os.remove(out)
                del running[pr]
    return results


def verdict(pages):
    n = len(pages)
    if n == 0:
        return "empty"
    c = Counter(p["quality"] for p in pages)
    textual = n - c["empty"] - c["sparse"]
    if c["empty"] / n > 0.6:
        return "scanned_or_empty"
    if textual == 0:
        return "scanned_or_empty"
    if c["indic_unicode"] / textual > 0.5:
        return "indic_unicode"
    if (c["ok"] + c["numeric"] + c["table"]) / textual >= 0.7:
        return "good"
    if c["garbled"] / textual > 0.5:
        return "garbled_legacy_font"
    return "mixed"


# ----------------------------------------------------------------------------- driver
def load_status(out_dir):
    st = {}
    p = os.path.join(out_dir, "status.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
                st[r["doc_id"]] = r
            except Exception:
                pass
    return st


def run(manifest_path, out_dir, tmp_dir, workers=8, procs=4, batch=120, limit=0, max_hours=10.5,
        keep_pdfs_dir=None, retry_failed=False, only_ids=None):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    docs = json.load(open(manifest_path, encoding="utf-8"))["docs"]
    if only_ids:
        docs = [d for d in docs if d["doc_id"] in only_ids]
    status = load_status(out_dir)
    todo = [d for d in docs if d["doc_id"] not in status
            or (retry_failed and status[d["doc_id"]]["status"] not in TERMINAL)]
    if limit:
        todo = todo[:limit]
    print("manifest %d | already done %d | to do %d" % (len(docs), len(status), len(todo)))
    shard = len([f for f in os.listdir(out_dir) if f.startswith("pages_")])
    t0 = time.time()
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers * 2))
    done = 0
    total_status, total_pages, total_bytes = Counter(), 0, 0
    log_path = os.path.join(out_dir, "progress.log")
    bar = _bar(len(todo))

    def log(msg):
        line = "%s | %s" % (time.strftime("%H:%M:%S"), msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return line

    def snapshot(stage):
        el = max(1e-9, time.time() - t0)
        rate = done / el
        eta = (len(todo) - done) / rate / 60 if rate > 0 else float("nan")
        snap = {"stage": stage, "done": done, "total": len(todo), "elapsed_min": round(el / 60, 1),
                "docs_per_min": round(rate * 60, 1), "eta_min": round(eta, 1), "status": dict(total_status),
                "pages": total_pages, "downloaded_mb": total_bytes // 2 ** 20,
                "disk_free_gb": round(shutil.disk_usage(out_dir).free / 2 ** 30, 1)}
        json.dump(snap, open(os.path.join(out_dir, "progress.json"), "w"))
        return snap

    log("start: %d docs to do, workers=%d, procs=%d, batch=%d" % (len(todo), workers, procs, batch))
    for b in range(0, len(todo), batch):
        if (time.time() - t0) / 3600 > max_hours:
            msg = log("time budget reached; stopping cleanly (rerun to resume)")
            print(msg)
            break
        chunk = todo[b:b + batch]
        bar.set_description("download")
        res = [None] * len(chunk)
        live = Counter()
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(download, d, tmp_dir, sess): i for i, d in enumerate(chunk)}
            for fut in as_completed(futs):
                res[futs[fut]] = fut.result()
                live[fut.result()[0]] += 1
                bar.update(1)
                ok = total_status["ok"] + live["ok"]
                bad = sum(total_status.values()) - total_status["ok"] + sum(live.values()) - live["ok"]
                bar.set_postfix_str("ok=%d fail=%d last=%s" % (ok, bad, fut.result()[0]))
        bar.set_description("extract ")
        recs, paths = {}, []
        for d, (st, path, info) in zip(chunk, res):
            recs[d["doc_id"]] = {"doc_id": d["doc_id"], "status": st, "kind": d["kind"],
                                 "name": (d["names"] or [""])[0], "num_questions": d["num_questions"], **info}
            if path:
                paths.append(path)
        pages_out = []
        if paths:
            for doc_id, (pages, err) in extract_all(paths, procs).items():
                r = recs[doc_id]
                if err == "timeout":
                    r["status"], r["error"] = "extract_timeout", "killed after %ds" % EXTRACT_DOC_S
                elif err:
                    r["status"], r["error"] = "extract_error", err
                else:
                    r["n_pages"] = len(pages)
                    r["verdict"] = verdict(pages)
                    r["quality_counts"] = dict(Counter(p["quality"] for p in pages))
                    pages_out.extend(pages)
        for p in paths:
            if keep_pdfs_dir:
                os.makedirs(keep_pdfs_dir, exist_ok=True)
                shutil.move(p, os.path.join(keep_pdfs_dir, os.path.basename(p)))
            elif os.path.exists(p):
                os.remove(p)
        if pages_out:
            with gzip.open(os.path.join(out_dir, "pages_%05d.jsonl.gz" % shard), "wt", encoding="utf-8") as f:
                for p in pages_out:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            shard += 1
        with open(os.path.join(out_dir, "status.jsonl"), "a", encoding="utf-8") as f:
            for r in recs.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done += len(chunk)
        c = Counter(r["status"] for r in recs.values())
        total_status.update(c)
        total_pages += len(pages_out)
        total_bytes += sum(r.get("bytes", 0) for r in recs.values())
        snap = snapshot("batch done")
        failed = [(r["doc_id"][:10], r["status"], r["num_questions"]) for r in recs.values() if r["status"] != "ok"]
        msg = log("batch %d/%d | %s | pages +%d | %.1f docs/min | ETA %.0f min | %d MB | disk free %.1f GB%s" % (
            done, len(todo), dict(c), len(pages_out), snap["docs_per_min"], snap["eta_min"],
            snap["downloaded_mb"], snap["disk_free_gb"],
            " | failed(id,status,nq): %s" % failed[:5] if failed else ""))
        tqdm_write(msg)
    bar.close()
    snapshot("finished")
    return summarize(out_dir)


def _bar(total):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, unit="doc", dynamic_ncols=True)
    except Exception:
        class _B:  # minimal fallback without tqdm
            def update(self, n=1): pass
            def set_description(self, s): pass
            def set_postfix_str(self, s): pass
            def close(self): pass
        return _B()


def tqdm_write(msg):
    try:
        from tqdm.auto import tqdm
        tqdm.write(msg)
    except Exception:
        print(msg)


def summarize(out_dir):
    st = load_status(out_dir)
    path = os.path.join(out_dir, "doc_summary.csv")
    cols = ["doc_id", "status", "verdict", "n_pages", "num_questions", "bytes", "name", "filename", "kind"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in st.values():
            w.writerow(r)
    print("status:", dict(Counter(r["status"] for r in st.values())))
    print("verdict:", dict(Counter(r.get("verdict", "-") for r in st.values() if r["status"] == "ok")))
    return st
