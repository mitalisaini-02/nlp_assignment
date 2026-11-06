"""Diagnostic A (CPU): can PDF retrieval be fixed, and would PDF excerpts add anything beyond the neighbour answers?

Same evaluation questions as Phase 2 (dev + test rows), same gold definition (retrieval.chunk_gold).
Methods compared (all as ranked lists of distinct (doc, page)):
  global_rrf     the stored Phase 2 chunk hits (BM25 + bge-small RRF over all 336k chunks)
  propagate      pages cited by the nearest training questions, in neighbour order
  scoped_bm25    BM25 over only the chunks of documents cited by the top-K neighbours
  global_rerank  global hits, pages of neighbour-cited documents moved first
  prop+scoped    propagated pages first, then scoped BM25 pages
Plus "support": share of the reference answer's content words / agronomic facts that appear in the text the model
would see (top-2 pages, the 2 neighbour answers, or both).

Run:  C:/nlp_data/venv/Scripts/python.exe -I src/diag_retrieval.py
"""
import csv
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retrieval as R  # noqa: E402
from bm25 import BM25  # noqa: E402
from eval_harness import extract_facts  # noqa: E402

PROJ = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(PROJ, "data", "processed", "train_processed.csv")
NN_PATH = "C:/Users/ARTH/Downloads/question_neighbours.json.gz"
HITS_PATH = "C:/Users/ARTH/Downloads/chunk_hits.json.gz"
CHUNKS = "C:/nlp_data/phase1_work/phase1/chunks.jsonl.gz"
OUT = "C:/nlp_data/diag"
K_NB = 5          # neighbours used for propagation / scoping
MS = (1, 2, 5, 10, 20)
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10 ** 9)
t0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


def jload(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
qid2row = {r["question_id"]: r for r in rows}
NN, HITS = jload(NN_PATH), jload(HITS_PATH)
ev = [r for r in rows if r["split"] in ("dev", "test")]
log("eval questions (dev+test):", len(ev))

# ---------------------------------------------------------------- pass 1: chunk ids + BM25 matrix (streamed)
ch_doc, ch_page = [], []


def stream():
    with gzip.open(CHUNKS, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ch_doc.append(d["doc_id"])
            ch_page.append(int(d["page"]))
            yield d["text"]


bm = BM25().fit(stream())
X = bm.W.T.tocsr()            # (n_chunks, vocab)
del bm.W
log("chunks:", len(ch_doc), "| BM25 fitted")
chunk_pages, chunk_docs = set(zip(ch_doc, ch_page)), set(ch_doc)
doc2chunks = defaultdict(list)
for i, d in enumerate(ch_doc):
    doc2chunks[d].append(i)


def cites(qid):
    """[(doc, [pages])] cited by a training row, zoho docs only."""
    return [(c["doc_id"], c.get("pages", [])) for c in json.loads(qid2row[qid]["citations_json"]) if c.get("kind") == "zoho"]


def dedupe_pages(chunk_ids):
    out, seen = [], set()
    for c in chunk_ids:
        k = (ch_doc[c], ch_page[c])
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# ---------------------------------------------------------------- build ranked page lists per method
lists = {m: {} for m in ("global_rrf", "propagate", "scoped_bm25", "global_rerank", "prop+scoped")}
gold = {}
nb_docs_cov, nb_pages_cov = [], []
for r in ev:
    q, split = r["question_id"], r["split"]
    gp, gd = R.chunk_gold(r["citations_json"], chunk_pages, chunk_docs)
    gold[q] = (gp, gd)
    nids = [n[0] for n in NN[split].get(q, [])[:K_NB]]
    # propagation: cited pages that exist as chunks, neighbour order; docs cited without pages -> nothing here
    prop, pdocs = [], []
    for nid in nids:
        for d, pages in cites(nid):
            if d in chunk_docs and d not in pdocs:
                pdocs.append(d)
            for p in pages:
                if (d, p) in chunk_pages and (d, p) not in prop:
                    prop.append((d, p))
    lists["propagate"][q] = prop
    g = dedupe_pages(HITS[split].get(q, []))
    lists["global_rrf"][q] = g
    pd_set = set(pdocs)
    lists["global_rerank"][q] = [p for p in g if p[0] in pd_set] + [p for p in g if p[0] not in pd_set]
    # scoped BM25 inside neighbour-cited docs
    cand = [c for d in pdocs for c in doc2chunks[d]]
    if cand:
        qv = bm.cv.transform([r["norm_q"]]).tocsr()
        qv.data[:] = 1.0
        s = np.asarray((X[cand] @ qv.T).todense()).ravel()
        order = np.argsort(-s)[:200]
        lists["scoped_bm25"][q] = dedupe_pages([cand[i] for i in order if s[i] > 0])
    else:
        lists["scoped_bm25"][q] = []
    ps = list(prop)
    for p in lists["scoped_bm25"][q]:
        if p not in ps:
            ps.append(p)
    lists["prop+scoped"][q] = ps
    if gd:
        nb_docs_cov.append(bool(gd & pd_set))
    if gp:
        nb_pages_cov.append(bool(gp & set(prop)))
log("lists built")

recall = {}
for name, L in lists.items():
    out = {}
    for m in MS:
        pg = [any(p in gold[q][0] for p in L[q][:m]) for q in L if gold[q][0]]
        dc = [any(p[0] in gold[q][1] for p in L[q][:m]) for q in L if gold[q][1]]
        out["page_hit@%d" % m], out["doc_hit@%d" % m] = float(np.mean(pg)), float(np.mean(dc))
    out["n_page_q"], out["n_doc_q"] = len(pg), len(dc)
    out["mean_list_len"] = float(np.mean([len(L[q]) for q in L]))
    out["empty_lists"] = int(sum(1 for q in L if not L[q]))
    recall[name] = out
ceiling = {"top%d_neighbours_cite_a_gold_doc" % K_NB: float(np.mean(nb_docs_cov)),
           "top%d_neighbours_cite_a_gold_page" % K_NB: float(np.mean(nb_pages_cov))}

print("\nPAGE / DOC HIT (distinct pages; n_page_q=%d, n_doc_q=%d)" % (recall["global_rrf"]["n_page_q"], recall["global_rrf"]["n_doc_q"]))
print("%-14s " % "" + " ".join("pg@%-3d" % m for m in MS) + "  " + " ".join("doc@%-3d" % m for m in MS) + "  avg_len empty")
for n, o in recall.items():
    print("%-14s " % n + " ".join("%.3f " % o["page_hit@%d" % m] for m in MS) + "  " +
          " ".join("%.3f  " % o["doc_hit@%d" % m] for m in MS) + "  %6.1f %5d" % (o["mean_list_len"], o["empty_lists"]))
print("ceiling:", ceiling)

# ---------------------------------------------------------------- pass 2: page texts for the support measurement
need = set()
for q in gold:
    need |= gold[q][0]
    for n in lists:
        need.update(lists[n][q][:2])
page_text = defaultdict(list)
with gzip.open(CHUNKS, "rt", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        k = (d["doc_id"], int(d["page"]))
        if k in need:
            page_text[k].append(d["text"])
page_text = {k: " ".join(v) for k, v in page_text.items()}
log("page texts loaded:", len(page_text))

TOK = re.compile(r"[a-z][a-z0-9\-]+|\d+(?:\.\d+)?")


def content(t):
    return {w for w in TOK.findall(t.lower()) if len(w) > 2 and w not in ENGLISH_STOP_WORDS}


def trim_words(t, n):
    return " ".join(t.split()[:n])


support = defaultdict(lambda: {"word_recall": [], "fact_recall": [], "words_shown": []})
for r in ev:
    q, split = r["question_id"], r["split"]
    ref = r["final_answer"]
    rw, rf = content(ref), extract_facts(ref)
    nids = [n[0] for n in NN[split].get(q, [])[:2]]
    nb_text = " ".join(trim_words(qid2row[n]["final_answer"], 290) for n in nids)  # ~380 tokens each, as in the prompt
    views = {"neighbour answers (current prompt)": nb_text,
             "gold cited pages (top-2)": " ".join(page_text.get(p, "") for p in list(gold[q][0])[:2])}
    for n in lists:
        views["%s top-2 pages" % n] = " ".join(page_text.get(p, "") for p in lists[n][q][:2])
    views["neighbours + propagate top-2"] = nb_text + " " + views["propagate top-2 pages"]
    views["neighbours + prop+scoped top-2"] = nb_text + " " + views["prop+scoped top-2 pages"]
    for name, txt in views.items():
        if name.startswith("gold") and not gold[q][0]:
            continue
        cw = content(txt)
        S = support[name]
        S["word_recall"].append(len(rw & cw) / max(1, len(rw)))
        if rf:
            tf = extract_facts(txt)
            S["fact_recall"].append(len(rf & tf) / len(rf))
        S["words_shown"].append(len(txt.split()))
support_out = {n: {"n": len(v["word_recall"]), "content_word_recall": float(np.mean(v["word_recall"])),
                   "fact_recall": float(np.mean(v["fact_recall"])) if v["fact_recall"] else None,
                   "mean_words_shown": float(np.mean(v["words_shown"]))} for n, v in support.items()}
print("\nSUPPORT: share of the reference answer's content words / facts present in what the model is shown")
print("%-40s %6s %10s %10s %8s" % ("view", "n", "word_rec", "fact_rec", "words"))
for n, v in support_out.items():
    print("%-40s %6d %10.3f %10.3f %8.0f" % (n, v["n"], v["content_word_recall"], v["fact_recall"] or 0, v["mean_words_shown"]))

json.dump({"eval": "dev+test rows (Phase 2 set)", "k_neighbours": K_NB, "recall": recall, "ceiling": ceiling,
           "support": support_out}, open(OUT + "/diag_retrieval.json", "w"), indent=1)
log("wrote", OUT + "/diag_retrieval.json")
