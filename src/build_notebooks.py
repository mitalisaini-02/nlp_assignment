"""Generate Kaggle-importable notebooks into notebooks/. Embeds src/pdf_pipeline.py so the notebook is self-contained.

    python src/build_notebooks.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build_download_nb():
    pipeline_src = open(os.path.join(ROOT, "src", "pdf_pipeline.py"), encoding="utf-8").read()
    cells = [
        md("""
# 01 - Download source PDFs and extract page text

> ## READ FIRST - files from an interactive session are LOST when the session ends
> `/kaggle/working` is only kept if you press **Save Version -> Save & Run All (Commit)**. That runs in the background on
> Kaggle (no browser tab needed, survives disconnects/idle timeouts) and stores `pdf_text/` in the version's **Output** tab.
> Do **not** run the full download interactively. Afterwards click **New Dataset** on the Output tab to keep it permanently.

**Before running**
1. Right panel -> *Session options* -> **Internet: On**. Accelerator can be **None** (CPU is enough).
2. *Add Input* -> upload a Kaggle Dataset containing `sources_manifest.json`
   (from `data/processed/` of the project).
3. To **resume** after a stopped/timed-out run: attach the previous version's output as an extra input;
   the notebook copies its `pdf_text/` folder and skips finished documents.

**What it does**: for each document in the manifest (most-cited first) it downloads the PDF, extracts per-page text with
PyMuPDF, scores extraction quality (flags scanned / legacy-font garbled / Indic-Unicode pages), writes compressed text shards to
`/kaggle/working/pdf_text/`, and **deletes the PDF** (Kaggle's working dir is capped at 20 GB).
"""),
        code("!pip install -q pymupdf"),
        code("%%writefile pdf_pipeline.py\n" + pipeline_src),
        code('''
import glob, os, shutil, json, importlib
import pdf_pipeline as pp
importlib.reload(pp)

cands = glob.glob("/kaggle/input/**/sources_manifest.json", recursive=True)
assert cands, "Attach the dataset that contains sources_manifest.json"
MANIFEST = cands[0]
OUT_DIR = "/kaggle/working/pdf_text"
TMP_DIR = "/tmp/pdfs"

# resume from a previous run's output if it was attached as an input
prev = glob.glob("/kaggle/input/**/status.jsonl", recursive=True)  # works whether files sit at the dataset root or in pdf_text/
if prev and not os.path.exists(os.path.join(OUT_DIR, "status.jsonl")):
    shutil.copytree(os.path.dirname(prev[0]), OUT_DIR, dirs_exist_ok=True)
    print("resumed from", os.path.dirname(prev[0]), "->", len(open(OUT_DIR + "/status.jsonl").readlines()), "docs already done")
else:
    print("no previous output found: starting from scratch")
print("manifest:", MANIFEST, "| docs:", len(json.load(open(MANIFEST))["docs"]))
'''),
        md("## Smoke test: 6 docs, separate output dir (checks internet + endpoint + extraction)"),
        code('''
SMOKE = "/kaggle/working/_smoke"
shutil.rmtree(SMOKE, ignore_errors=True)
pp.run(MANIFEST, SMOKE, TMP_DIR, workers=6, procs=2, batch=6, limit=6)
import gzip
for line in gzip.open(sorted(glob.glob(SMOKE + "/pages_*.jsonl.gz"))[0], "rt"):
    p = json.loads(line)
    if p["quality"] in ("ok", "indic_unicode"):
        print(p["doc_id"], "page", p["page"], p["script"], p["quality"], "\\n", p["text"][:300]); break
'''),
        md("""
## Full run
`max_hours` stops cleanly before Kaggle's 12h session limit; re-run (with this version's output attached as input) to continue.
If many docs fail with `http_429`/timeouts, lower `workers`.
"""),
        code('''
shutil.rmtree(SMOKE, ignore_errors=True)
st = pp.run(MANIFEST, OUT_DIR, TMP_DIR, workers=8, procs=4, batch=120, max_hours=10.0, retry_failed=True)
'''),
        md("## Report: how much of the citation mass is usable?"),
        code('''
import csv, collections as C
rows = list(csv.DictReader(open(OUT_DIR + "/doc_summary.csv", encoding="utf-8")))
tot = sum(int(r["num_questions"] or 0) for r in rows) or 1
by = C.defaultdict(lambda: [0, 0])
for r in rows:
    k = r["verdict"] if r["status"] == "ok" else r["status"]
    by[k][0] += 1
    by[k][1] += int(r["num_questions"] or 0)
print("%-24s %6s %14s" % ("outcome", "docs", "citation share"))
for k, (n, q) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print("%-24s %6d %13.1f%%" % (k, n, 100 * q / tot))
print("\\noutput size (MB):", sum(os.path.getsize(f) for f in glob.glob(OUT_DIR + "/*")) // 2**20)
'''),
        md("""
## After it finishes
*Save Version -> Save & Run All (Commit)*, then on the output tab choose **New Dataset** to publish `pdf_text/`
(shards `pages_*.jsonl.gz`, `status.jsonl`, `doc_summary.csv`). Phase 1 step 6-9 (chunking, translation, indexing) read from that dataset.
"""),
    ]
    return notebook(cells)


def build_phase0_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 00 - Phase 0: data foundation

**Before running**: *Add Input* -> a Kaggle Dataset containing `training_set_15000.csv` and `unseen_set_1000.csv`.
Internet and GPU are not needed.

Produces (in `/kaggle/working/processed/`): `train_processed.csv` (group ids, medoid flag, split, parsed citations),
`test_processed.csv` (overlap with train flagged), `split.json` (IDs of train/dev/test - **keep this, the report must reproduce it exactly**),
`phase0_report.json`. The split is by *question group*, seed 42, so no question text appears in both train and test.
"""),
        code("!pip install -q rouge-score sacrebleu"),
        code("%%writefile sources.py\n" + src("sources.py")),
        code("%%writefile phase0_prepare.py\n" + src("phase0_prepare.py")),
        code("%%writefile eval_harness.py\n" + src("eval_harness.py")),
        code('''
import glob, os
hit = glob.glob("/kaggle/input/**/training_set_15000.csv", recursive=True)
assert hit, "Attach the dataset containing training_set_15000.csv and unseen_set_1000.csv"
os.environ["NLP_DATA_DIR"] = os.path.dirname(hit[0])
os.environ["NLP_OUT_DIR"] = "/kaggle/working/processed"
print("data dir:", os.environ["NLP_DATA_DIR"])
'''),
        md("## Run Phase 0 (grouping, medoids, grouped split, citation parsing; ~1-2 min)"),
        code("import phase0_prepare\nphase0_prepare.main()"),
        md("## Sanity checks and evaluation-harness self-test"),
        code('''
import csv, json
csv.field_size_limit(10**9)
P = "/kaggle/working/processed/"
rows = list(csv.DictReader(open(P + "train_processed.csv", encoding="utf-8")))
sp = {s: {r["norm_q"] for r in rows if r["split"] == s} for s in ("train", "dev", "test")}
assert not (sp["test"] & (sp["train"] | sp["dev"])), "leak"
assert not (sp["dev"] & sp["train"]), "leak"
print({s: sum(r["split"] == s for r in rows) for s in sp}, "- no question leaks across splits")
print("medoids per group ok:", len({r["group_id"] for r in rows if r["is_medoid"] == "1"}) == len({r["group_id"] for r in rows}))
from eval_harness import evaluate
test = [r for r in rows if r["split"] == "test"]
refs = {r["question_id"]: r["final_answer"] for r in test}
print("self-score (should be 1.0):", {k: round(v, 3) for k, v in evaluate(refs, refs).items() if k in ("rouge1", "rougeL", "bleu", "fact_f1")})
print(json.dumps(json.load(open(P + "phase0_report.json")), indent=1))
'''),
        md("""
## Save outputs
*Save Version -> Save & Run All*, then publish `processed/` as a Kaggle Dataset (e.g. `nlp-qna-processed`). Later notebooks
attach it for the train/dev/test splits. Also copy `eval_harness.py` from the Output tab - every later notebook reuses it.
"""),
    ]
    return notebook(cells)


def build_phase1_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 02 - Phase 1: quality triage and chunking of the extracted PDF text

> Run this with **Save Version -> Save & Run All (Commit)**, not interactively: interactive output is lost when the session ends.

**Page-quality labels are recomputed here from the saved text** (using the same `page_stats` / `verdict` code as the
downloader). Labels stored by the download run are only used to report how many differ - an earlier download run was
produced with a broken classifier line and labelled almost everything as garbled, but the extracted text itself is fine.

**Inputs (Add Input)**
1. the dataset made from `pdf_text/` (contains `status.jsonl` and `pages_*.jsonl.gz`)
2. *recommended*: the Phase 0 dataset (contains `train_processed.csv`) - used to measure how much of what the training
   answers cite is actually available as text.

Internet and GPU are not needed (CPU only, uses all cores).

**Outputs** (`/kaggle/working/phase1/`): `chunks.jsonl.gz` (one retrieval chunk per line, each with exact `doc_id` + `page`),
`doc_verdicts.csv`, `phase1_report.json`.
"""),
        code("%%writefile pdf_pipeline.py\n" + src("pdf_pipeline.py")),
        code("%%writefile chunking.py\n" + src("chunking.py")),
        code("%%writefile phase1_worker.py\n" + src("phase1_worker.py")),
        code('''
import glob, os, json, gzip, csv, time, collections as C
from multiprocessing import Pool
import sys
_missing = [f for f in ("pdf_pipeline.py", "chunking.py", "phase1_worker.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - run the %%writefile cells at the top first (use Run All, do not start from a lower cell)"
sys.path.insert(0, os.getcwd())
import phase1_worker as w

hit = glob.glob("/kaggle/input/**/status.jsonl", recursive=True)
assert hit, "attach the dataset made from pdf_text/"
PDF_TEXT = os.path.dirname(hit[0])
proc = glob.glob("/kaggle/input/**/train_processed*.csv", recursive=True)
if not proc:  # show what IS attached, so a wrong name / wrong dataset is obvious
    print("train_processed*.csv not found. Files under /kaggle/input (first 40):")
    for i, (dp, dn, fn) in enumerate(os.walk("/kaggle/input")):
        for x in fn[:6]:
            print("  ", os.path.join(dp, x))
        if i > 40:
            break
OUT = "/kaggle/working/phase1"
os.makedirs(OUT, exist_ok=True)
NPROC = max(1, (os.cpu_count() or 2))
print("pdf text:", PDF_TEXT, "| phase0 file:", proc[0] if proc else "NOT attached (coverage section will be skipped)", "| procs:", NPROC)

st = {}                                   # last record per doc wins (retries append a second record)
for line in open(PDF_TEXT + "/status.jsonl", encoding="utf-8"):
    try:
        r = json.loads(line); st[r["doc_id"]] = r
    except Exception:
        pass
print("docs with a status record:", len(st))
'''),
        md("## 1. Recompute page quality and chunk usable documents\nOne pass over the shards; each document is re-labelled and chunked in a worker process."),
        code('''
# tolerate: shards in a subfolder, or decompressed to plain .jsonl by the dataset upload
shards = sorted(glob.glob("/kaggle/input/**/pages_*.jsonl*", recursive=True))
if not shards:
    print("NO page shards found. Contents of", PDF_TEXT, "->", sorted(os.listdir(PDF_TEXT))[:60])
assert shards, "the dataset has status.jsonl but no pages_*.jsonl(.gz) files - re-upload the shards (see the listing above)"
print("shards found:", len(shards), "| first:", shards[0])

def open_shard(f):
    return gzip.open(f, "rt", encoding="utf-8") if f.endswith(".gz") else open(f, encoding="utf-8")

seen, dup = set(), 0
verdicts, stored_vs_new = {}, C.Counter()
chunk_pages, docs_with_chunks = set(), set()
per_verdict, per_script, words, n_chunks = C.Counter(), C.Counter(), 0, 0
samples = C.defaultdict(list)
t0 = time.time()
with gzip.open(OUT + "/chunks.jsonl.gz", "wt", encoding="utf-8") as out, Pool(NPROC) as pool:
    for i, f in enumerate(shards):
        by_doc = C.defaultdict(list)
        for line in open_shard(f):
            p = json.loads(line)
            key = (p["doc_id"], p["page"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            by_doc[p["doc_id"]].append(p)
        for doc, verdict, n_pages, qc, chunks, sample in pool.imap_unordered(w.requalify_and_chunk, by_doc.items(), chunksize=4):
            verdicts[doc] = verdict
            stored_vs_new[(st.get(doc, {}).get("verdict"), verdict)] += 1
            if sample and len(samples[verdict]) < 5:
                samples[verdict].append((doc, sample))
            for c in chunks:
                out.write(json.dumps(c, ensure_ascii=False) + "\\n")
                n_chunks += 1
                words += c["n_words"]
                per_verdict[verdict] += 1
                per_script[c["script"]] += 1
                chunk_pages.add((c["doc_id"], c["page"]))
                docs_with_chunks.add(doc)
        if (i + 1) % 5 == 0 or i + 1 == len(shards):
            print("shard %d/%d | docs %d | chunks %d | %.1f min" % (i + 1, len(shards), len(verdicts), n_chunks, (time.time() - t0) / 60))
print("chunks:", n_chunks, "| avg words/chunk: %.0f" % (words / max(1, n_chunks)), "| duplicate pages skipped:", dup)
print("chunks per verdict:", dict(per_verdict))
print("chunks per script :", dict(per_script))
print("documents with at least one chunk:", len(docs_with_chunks), "of", len(st))
'''),
        md("## 2. Quality triage - how much of the citation mass is usable?\n`citation share` weights each document by the number of training questions that cite it. Verdicts are the **recomputed** ones."),
        code('''
def final_outcome(doc, r):
    return verdicts.get(doc, r.get("verdict")) if r["status"] == "ok" else r["status"]

tot = sum(int(r.get("num_questions") or 0) for r in st.values()) or 1
by = C.defaultdict(lambda: [0, 0])
for d, r in st.items():
    k = final_outcome(d, r)
    by[k][0] += 1
    by[k][1] += int(r.get("num_questions") or 0)
print("%-24s %6s %15s" % ("outcome", "docs", "citation share"))
for k, (n, q) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print("%-24s %6d %14.1f%%" % (k, n, 100 * q / tot))
print("\\nstored label -> recomputed label (docs):")
for (a, b), n in stored_vs_new.most_common(10):
    print("  %-22s -> %-22s %5d" % (a, b, n))
'''),
        md("## 3. What gets excluded - sample pages (garbled / mixed documents)"),
        code('''
for v, items in samples.items():
    for doc, s in items:
        print("=" * 90)
        print("verdict:", v, "|", (st[doc].get("filename") or st[doc].get("name") or "")[:70], "| page", s["page"], "| sw", s["sw"], "| cited by", st[doc].get("num_questions"), "questions")
        print(repr(s["text"]))
'''),
        md("## 4. Does the extracted text cover what the training answers cite?"),
        code('''
rep = {"chunks": n_chunks, "docs_with_chunks": len(docs_with_chunks), "docs_total": len(st), "duplicate_pages_skipped": dup,
       "chunks_per_verdict": dict(per_verdict), "chunks_per_script": dict(per_script),
       "outcome_docs_and_citations": {k: v for k, v in by.items()}}
if proc:
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(proc[0], encoding="utf-8")))
    stats = C.defaultdict(lambda: C.Counter())
    for r in rows:
        cits = [c for c in json.loads(r["citations_json"]) if c["kind"] == "zoho"]
        s = stats[r["split"]]
        s["rows"] += 1
        s["rows_with_citation"] += bool(cits)
        s["rows_with_usable_doc"] += any(c["doc_id"] in docs_with_chunks for c in cits)
        for c in cits:
            for pg in c["pages"]:
                s["cited_pages"] += 1
                s["cited_pages_with_chunk"] += (c["doc_id"], pg) in chunk_pages
    print("%-6s %6s %14s %20s %22s" % ("split", "rows", "with citation", "with a usable doc", "cited pages available"))
    for sp in ("train", "dev", "test"):
        s = stats[sp]
        print("%-6s %6d %13.1f%% %19.1f%% %21.1f%%" % (sp, s["rows"], 100 * s["rows_with_citation"] / s["rows"],
              100 * s["rows_with_usable_doc"] / s["rows"], 100 * s["cited_pages_with_chunk"] / max(1, s["cited_pages"])))
    rep["train_coverage"] = {sp: dict(stats[sp]) for sp in stats}
json.dump(rep, open(OUT + "/phase1_report.json", "w"), indent=1)

with open(OUT + "/doc_verdicts.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["doc_id", "status", "stored_verdict", "verdict", "num_questions", "n_pages", "has_chunks", "filename", "name"])
    for d, r in st.items():
        wr.writerow([d, r["status"], r.get("verdict", ""), final_outcome(d, r), r.get("num_questions", ""), r.get("n_pages", ""),
                     int(d in docs_with_chunks), r.get("filename", ""), r.get("name", "")])
print("saved:", os.listdir(OUT))
'''),
        md("""
## After it finishes
Save Version -> Save & Run All, then on the Output tab press **New Dataset** (e.g. `nlp-qna-chunks`). The retrieval notebook
(Phase 2) attaches this dataset. Keep the printed tables (quality triage, coverage) - they go in the report.
"""),
    ]
    return notebook(cells)


def build_phase2_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 03 - Phase 2: retrieval layer and the nearest-neighbour baseline (B0)

> Run with **Save Version -> Save & Run All (Commit)** (interactive output is lost when the session ends).
> Settings: **Accelerator = GPU T4** (one is enough), **Internet = On** (downloads the embedding models from Hugging Face).

**Inputs (Add Input)**
1. the Phase 0 dataset: `train_processed.csv` and `test_processed.csv` (the latter is the unseen 1,000-question set)
2. the chunks dataset: `chunks.jsonl.gz` (from Phase 1)

**What it builds**
* **Index A** - training questions: BM25 + `bge-m3` dense, fused with reciprocal-rank fusion (RRF). Tuned on the dev split.
  Pool = the 12,825 *train* rows only (so dev/test measurements are leak-free); the unseen set uses all 15,000 rows.
* **Index B** - PDF chunks: BM25 + `bge-small-en-v1.5` dense, fused with RRF. Quality is measured against the pages the
  training answers actually cite (page-hit / doc-hit @ m).
* **B0 baseline** - answer a question with the answer of its nearest training question (and a medoid-of-top-5 variant),
  scored on the 1,500 hold-out rows with the shared evaluation harness.
* **Artifacts for Phase 4**: top-10 neighbour questions and top-20 chunks for every train/dev/test/unseen question, plus all embeddings.
"""),
        code("!pip install -q rouge-score sacrebleu sentence-transformers"),
        code("%%writefile eval_harness.py\n" + src("eval_harness.py")),
        code("%%writefile bm25.py\n" + src("bm25.py")),
        code("%%writefile encoders.py\n" + src("encoders.py")),
        code("%%writefile retrieval.py\n" + src("retrieval.py")),
        md("## 0. Load data"),
        code('''
import glob, os, sys, json, gzip, csv, time, collections as C
import numpy as np
csv.field_size_limit(10 ** 9)
_missing = [f for f in ("eval_harness.py", "bm25.py", "encoders.py", "retrieval.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - run the %%writefile cells at the top first (use Run All, do not start from a lower cell)"
sys.path.insert(0, os.getcwd())

def find(pattern):
    hit = glob.glob("/kaggle/input/**/" + pattern, recursive=True)
    if not hit:  # show what IS attached, so a wrong name / missing dataset is obvious
        print("not found:", pattern, "- files under /kaggle/input (first 40):")
        for i, (dp, dn, fn) in enumerate(os.walk("/kaggle/input")):
            for x in fn[:8]:
                print("  ", os.path.join(dp, x))
            if i > 40:
                break
    assert hit, "attach a dataset containing " + pattern
    return hit[0]

# chunks.jsonl may have been decompressed to plain .jsonl when the dataset was created, so accept both
TRAIN_CSV, UNSEEN_CSV, CHUNKS = find("train_processed*.csv"), find("test_processed*.csv"), find("chunks.jsonl*")
OUT = "/kaggle/working/phase2"
os.makedirs(OUT, exist_ok=True)
QLIMIT = int(os.environ.get("P2_QLIMIT", 0))            # debug hooks only; 0 = use everything
CHUNK_LIMIT = int(os.environ.get("P2_CHUNK_LIMIT", 0))
Q_ENCODER, C_ENCODER = "BAAI/bge-m3", "BAAI/bge-small-en-v1.5"
C_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEPTH, TOPN_Q, TOPN_C = 60, 10, 20
print(TRAIN_CSV, "|", UNSEEN_CSV, "|", CHUNKS)

rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
unseen = list(csv.DictReader(open(UNSEEN_CSV, encoding="utf-8")))
split = np.array([r["split"] for r in rows])
grp = np.array([r["group_id"] for r in rows])
norm = [r["norm_q"] for r in rows]
answers = [r["final_answer"] for r in rows]
qids = [r["question_id"] for r in rows]
ix = {s: np.where(split == s)[0] for s in ("train", "dev", "test")}
if QLIMIT:
    ix["dev"], ix["test"], ix["train"] = ix["dev"][:QLIMIT], ix["test"][:QLIMIT], ix["train"][:3 * QLIMIT]
pool_tr = np.where(split == "train")[0]            # the retrieval pool for dev / test / train queries
print({s: len(v) for s, v in ix.items()}, "| pool:", len(pool_tr), "| unseen:", len(unseen))

ch_doc, ch_page, ch_text = [], [], []
for line in (gzip.open(CHUNKS, "rt", encoding="utf-8") if CHUNKS.endswith(".gz") else open(CHUNKS, encoding="utf-8")):
    c = json.loads(line)
    ch_doc.append(c["doc_id"]); ch_page.append(c["page"]); ch_text.append(c["text"])
    if CHUNK_LIMIT and len(ch_text) >= CHUNK_LIMIT:
        break
print("chunks:", len(ch_text))
'''),
        md("## 1. Embeddings (GPU)\n`bge-m3` for question-to-question matching (multilingual - handles the Gurmukhi/Devanagari questions); `bge-small-en` for question-to-chunk matching (336k chunks - 17x cheaper)."),
        code('''
import encoders as enc
t0 = time.time()
all_q = [r["question"] for r in rows] + [r["question"] for r in unseen]
Qm = enc.encode(Q_ENCODER, all_q, batch_size=64)
print("question embeddings (m3):", Qm.shape, "%.1f min" % ((time.time() - t0) / 60))
Qc = enc.encode(C_ENCODER, all_q, prefix=C_QUERY_PREFIX, max_len=64)
Cemb = enc.encode(C_ENCODER, ch_text, batch_size=128, max_len=320)
print("chunk embeddings:", Cemb.shape, "%.1f min total" % ((time.time() - t0) / 60))
np.save(OUT + "/q_emb_m3.npy", Qm); np.save(OUT + "/q_emb_chunkspace.npy", Qc); np.save(OUT + "/chunk_emb.npy", Cemb)
Qm_rows, Qm_unseen = Qm[:len(rows)], Qm[len(rows):]
Qc_rows, Qc_unseen = Qc[:len(rows)], Qc[len(rows):]
'''),
        md("## 2. Index A - training questions: tune the fusion on the dev split\nObjective: ROUGE-L between the top-1 neighbour's answer and the true answer (answers truncated to 700 words for speed)."),
        code('''
from bm25 import BM25
import retrieval as R
bm_tr = BM25().fit([norm[i] for i in pool_tr])
D_tr = Qm_rows[pool_tr]
dev = ix["dev"]
b_i, _ = bm_tr.search([norm[i] for i in dev], k=DEPTH)
d_i, _ = R.dense_topk(Qm_rows[dev], D_tr, k=DEPTH)
gold_dev = [answers[i] for i in dev]

def top1_rougeL(ids):
    pairs = [(answers[pool_tr[c]] if c >= 0 else "", gold_dev[q]) for q, c in enumerate(ids[:, 0])]
    return float(np.mean(R.rougeL_pairs(pairs)))

rng = np.random.RandomState(0)
tune = {"random train answer (floor)": float(np.mean(R.rougeL_pairs(
            [(answers[pool_tr[rng.randint(len(pool_tr))]], g) for g in gold_dev]))),
        "BM25 only": top1_rougeL(b_i), "dense (bge-m3) only": top1_rougeL(d_i)}
fused = {}
for rk in (20, 60):
    for wd in (1.0, 2.0):
        ids, _ = R.rrf([b_i, d_i], [1.0, wd], k=rk, topn=10)
        name = "RRF k=%d  w_dense=%.0f" % (rk, wd)
        tune[name] = top1_rougeL(ids); fused[name] = (rk, wd, ids)
best_name = max(fused, key=lambda n: tune[n])
BEST_K, BEST_W, best_ids = fused[best_name]
print("%-30s %s" % ("retriever (dev, %d questions)" % len(dev), "top-1 ROUGE-L"))
for n, v in tune.items():
    print("%-30s %.4f%s" % (n, v, "   <- chosen" if n == best_name else ""))
# how much headroom is there if we could pick the best of the top-5 neighbours?
pairs = [(answers[pool_tr[c]], gold_dev[q]) for q in range(len(dev)) for c in best_ids[q, :5] if c >= 0]
sc5 = R.rougeL_pairs(pairs); i = 0; orc = []
for q in range(len(dev)):
    m = sum(1 for c in best_ids[q, :5] if c >= 0); orc.append(max(sc5[i:i + m]) if m else 0.0); i += m
print("oracle best-of-5 neighbours: %.4f" % np.mean(orc))
'''),
        md("## 3. Index A - neighbours for every query set (saved for Phase 4)\ntrain queries exclude their own group; dev/test use the train pool; the unseen set uses all 15,000 questions."),
        code('''
bm_all = BM25().fit(norm)

def neighbours(q_norm, q_dense, bm, D, pool_rows, q_group=None):
    b, _ = bm.search(q_norm, k=DEPTH)
    d, _ = R.dense_topk(q_dense, D, k=DEPTH)
    ids, sc = R.rrf([b, d], [1.0, BEST_W], k=BEST_K, topn=DEPTH)
    if q_group is not None:
        return R.drop_groups(ids, sc, grp[pool_rows], q_group, TOPN_Q)
    return ids[:, :TOPN_Q], sc[:, :TOPN_Q]

NN = {}
for s in ("train", "dev", "test"):
    q = ix[s]
    NN[s] = neighbours([norm[i] for i in q], Qm_rows[q], bm_tr, D_tr, pool_tr, grp[q] if s == "train" else None)
NN["xt"] = neighbours([r["norm_q"] for r in unseen], Qm_unseen, bm_all, Qm_rows, np.arange(len(rows)))
pool_of = {"train": pool_tr, "dev": pool_tr, "test": pool_tr, "xt": np.arange(len(rows))}
q_ids_of = {s: [qids[i] for i in ix[s]] for s in ("train", "dev", "test")}
q_ids_of["xt"] = [r["question_id"] for r in unseen]
out = {s: {q_ids_of[s][k]: [[qids[pool_of[s][c]], round(float(sc), 5)] for c, sc in zip(NN[s][0][k], NN[s][1][k]) if c >= 0]
           for k in range(len(q_ids_of[s]))} for s in NN}
with gzip.open(OUT + "/question_neighbours.json.gz", "wt", encoding="utf-8") as f:
    json.dump(out, f)
print({s: len(v) for s, v in out.items()}, "| example:", list(out["dev"].items())[0])
'''),
        md("## 4. Index B - PDF chunks: BM25 + dense, scored against the pages the training answers cite"),
        code('''
t0 = time.time()
bm_c = BM25().fit(ch_text)
print("chunk BM25 fitted: %.1f min" % ((time.time() - t0) / 60))
q_texts = [r["norm_q"] for r in rows] + [r["norm_q"] for r in unseen]
cb_i, _ = bm_c.search(q_texts, k=DEPTH)
cd_i, _ = R.dense_topk(np.concatenate([Qc_rows, Qc_unseen]), Cemb, k=DEPTH)
print("chunk retrieval for all %d queries: %.1f min" % (len(q_texts), (time.time() - t0) / 60))

chunk_pages, chunk_docs = set(zip(ch_doc, ch_page)), set(ch_doc)
ev = np.concatenate([ix["dev"], ix["test"]])
gp, gd = zip(*[R.chunk_gold(rows[i]["citations_json"], chunk_pages, chunk_docs) for i in ev])
variants = {"BM25 only": cb_i[ev], "dense (bge-small) only": cd_i[ev]}
cfused = {}
for wd in (1.0, 2.0):
    ids, _ = R.rrf([cb_i[ev], cd_i[ev]], [1.0, wd], k=60, topn=TOPN_C)
    variants["RRF w_dense=%.0f" % wd] = ids; cfused[wd] = ids
recall = {n: R.chunk_recall(h, gp, gd, ch_doc, ch_page) for n, h in variants.items()}
print("questions with a gold page: %d | with a gold doc: %d (of %d dev+test)" % (
    recall["BM25 only"]["n_page_q"], recall["BM25 only"]["n_doc_q"], len(ev)))
print("%-24s %s" % ("", "  ".join("page@%-2d doc@%-2d" % (m, m) for m in (1, 5, 10, 20))))
for n, r in recall.items():
    print("%-24s %s" % (n, "  ".join("%.2f   %.2f " % (r["page_hit@%d" % m], r["doc_hit@%d" % m]) for m in (1, 5, 10, 20))))
BEST_CW = max(cfused, key=lambda w: recall["RRF w_dense=%.0f" % w]["page_hit@10"])
print("chosen chunk fusion: RRF w_dense=%.0f" % BEST_CW)
'''),
        code('''
cids, _ = R.rrf([cb_i, cd_i], [1.0, BEST_CW], k=60, topn=TOPN_C)
hits = {"train": {}, "dev": {}, "test": {}, "xt": {}}
for s in ("train", "dev", "test"):
    for i in ix[s]:
        hits[s][qids[i]] = [int(c) for c in cids[i] if c >= 0]
for k, r in enumerate(unseen):
    hits["xt"][r["question_id"]] = [int(c) for c in cids[len(rows) + k] if c >= 0]
with gzip.open(OUT + "/chunk_hits.json.gz", "wt", encoding="utf-8") as f:
    json.dump(hits, f)   # chunk ids = line numbers (0-based) of chunks.jsonl.gz
print({s: len(v) for s, v in hits.items()})
'''),
        md("## 5. B0 baselines on the 1,500 hold-out rows\nB0a: answer = the nearest training question's answer. B0b: the medoid (highest mean ROUGE-L vs the others) of the 5 nearest answers."),
        code('''
from eval_harness import evaluate
import csv as _csv
te = ix["test"]
refs = {qids[i]: answers[i] for i in te}
ids_te = NN["test"][0]
b0a = {qids[i]: answers[pool_tr[ids_te[k, 0]]] if ids_te[k, 0] >= 0 else "" for k, i in enumerate(te)}
cand = [[answers[pool_tr[c]] for c in ids_te[k, :5] if c >= 0] or [""] for k in range(len(te))]
pick = R.mbr_pick(cand)
b0b = {qids[i]: cand[k][pick[k]] for k, i in enumerate(te)}
res = {"B0a nearest-neighbour answer": evaluate(b0a, refs), "B0b medoid of top-5": evaluate(b0b, refs)}
cols = ["rouge1", "rouge2", "rougeL", "bleu", "len_ratio_mean", "fact_recall", "fact_precision", "fact_f1"]
print("%-32s %s" % ("hold-out test (n=%d)" % len(te), "  ".join("%-9s" % c[:9] for c in cols)))
for n, m in res.items():
    print("%-32s %s" % (n, "  ".join("%-9.4f" % m[c] for c in cols)))
with open(OUT + "/b0_test_predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = _csv.writer(f); w.writerow(["question_id", "b0a_nearest", "b0b_medoid5"])
    for q in refs:
        w.writerow([q, b0a[q], b0b[q]])
'''),
        md("## 6. Save the report"),
        code('''
report = {"encoders": {"questions": Q_ENCODER, "chunks": C_ENCODER}, "rrf": {"k": BEST_K, "w_dense": BEST_W, "chunk_w_dense": BEST_CW},
          "dev_tuning_top1_rougeL": tune, "oracle_best_of_5": float(np.mean(orc)),
          "chunk_retrieval_recall": recall, "b0_holdout": res,
          "sizes": {"train_pool": len(pool_tr), "chunks": len(ch_text), "unseen": len(unseen)}}
json.dump(report, open(OUT + "/phase2_report.json", "w"), indent=1)
print(sorted(os.listdir(OUT)))
print("total MB:", sum(os.path.getsize(OUT + "/" + f) for f in os.listdir(OUT)) // 2 ** 20)
'''),
        md("""
## After it finishes
Output tab -> **New Dataset** (e.g. `nlp-qna-retrieval`). Phase 4 (training) attaches it for the neighbour lists, chunk hits and embeddings.
Keep the printed tables - the dev tuning table, the chunk-recall table and the B0 table go straight into the report.
"""),
    ]
    return notebook(cells)


def build_phase4_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 04 - Phase 4: QLoRA fine-tuning of the answer-writing model

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
> A commit run is limited to 12 h; the notebook stops training cleanly before that (`TRAIN_HOURS`).

**Inputs (Add Input):** the Phase 0 dataset (`train_processed.csv`), the Phase 2 output dataset (`question_neighbours.json`, `chunk_hits.json`),
and - only for the `full` mode - the chunks dataset (`chunks.jsonl`).

**Three variants (the ablation for the report)**

| mode | prompt | role in the report |
|---|---|---|
| `none` | question only | B2: plain fine-tuning, no retrieval |
| `nn` | question + 2 nearest training Q&A pairs | proposed model without PDFs |
| `full` | `nn` + source-page excerpts (pages cited by the neighbours) | proposed model with PDFs |

`RUN` below chooses which modes to train in this session. Two modes train **in parallel, one per GPU**. Suggested order:
first `["nn", "none"]`, then in a second session `["full", ...]`, and finally the winning mode on the full training set.

**Loss is computed on the answer tokens only.** 20% of examples get a distractor reference (RAFT-style) so the model learns to ignore irrelevant context.
"""),
        code("!pip install -q -U peft bitsandbytes accelerate huggingface_hub"),
        code("%%writefile prompts.py\n" + src("prompts.py")),
        code("%%writefile train_qlora.py\n" + src("train_qlora.py")),
        md("## 0. Configuration"),
        code('''
import glob, os, sys, json, gzip, csv, random, time, subprocess, collections as C
csv.field_size_limit(10 ** 9)
_missing = [f for f in ("prompts.py", "train_qlora.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - run the %%writefile cells at the top first (use Run All, do not start from a lower cell)"
sys.path.insert(0, os.getcwd())

import torch
if not os.environ.get("P4_EXTRA"):   # (the env var is a CPU smoke-test hook, never set on Kaggle)
    assert torch.cuda.is_available(), ("No GPU attached to this session. Right panel -> Session options -> Accelerator -> "
                                       "GPU T4 x2 (the session restarts), then Run All from the top.")
    print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

MODEL = os.environ.get("P4_MODEL", "Qwen/Qwen2.5-3B-Instruct")
RUN = ["nn", "none"]          # modes to train in this session (see the table above); later e.g. ["full"]
N_TRAIN = 4000                # training questions per run (same subset for every mode). Final run: 0 = all 12,825
EPOCHS = 1.0
MAX_LEN = 2560                # prompt + answer tokens
GRAD_ACCUM = 16               # effective batch = 16 examples
NOISE = 0.2                   # share of examples with a distractor reference
N_DEV = 60                    # dev questions used for the eval-loss curve
TRAIN_HOURS = 8.0             # wall-clock limit per run; the adapter is saved when it is reached
EXTRA = os.environ.get("P4_EXTRA", "").split()   # debug hook only (CPU smoke tests); empty on Kaggle

def find(pattern, required=True):
    hit = glob.glob("/kaggle/input/**/" + pattern, recursive=True)
    if not hit and required:
        print("not found:", pattern, "- files under /kaggle/input (first 40):")
        for i, (dp, dn, fn) in enumerate(os.walk("/kaggle/input")):
            for x in fn[:8]:
                print("  ", os.path.join(dp, x))
            if i > 40:
                break
        raise AssertionError("attach a dataset containing " + pattern)
    return hit[0] if hit else None

def op(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")

TRAIN_CSV, NN_PATH = find("train_processed*.csv"), find("question_neighbours.json*")
HITS_PATH, CHUNKS = find("chunk_hits.json*", required=False), find("chunks.jsonl*", required="full" in RUN)
WORK = "/kaggle/working"
print("modes:", RUN, "| model:", MODEL, "| N_TRAIN:", N_TRAIN or "all")
'''),
        md("## 1. Build the prompts (CPU)"),
        code('''
import prompts as P
rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
NN = json.load(op(NN_PATH))
hits = json.load(op(HITS_PATH)) if (HITS_PATH and "full" in RUN) else {}
page_text, chunk_text = {}, []
if "full" in RUN:
    for line in op(CHUNKS):
        c = json.loads(line)
        chunk_text.append(c["text"])
        page_text.setdefault((c["doc_id"], c["page"]), c["text"])   # first chunk of each page
    print("chunks:", len(chunk_text), "| pages with text:", len(page_text))
ctx = P.Context(rows, page_text, chunk_text)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)   # used to trim context by TOKENS so no prompt exceeds MAX_LEN

rng = random.Random(42)
train_rows = [r for r in rows if r["split"] == "train"]
dev_rows = [r for r in rows if r["split"] == "dev"][:N_DEV]
sub = train_rows if not N_TRAIN else rng.sample(train_rows, min(N_TRAIN, len(train_rows)))
pool_qids = [r["question_id"] for r in train_rows]
print("train subset:", len(sub), "| dev:", len(dev_rows))

def build(mode, items, split, noise):
    r = random.Random(1)
    out = []
    for row in items:
        q = row["question_id"]
        out.append(P.make_example(ctx, mode, q, row["question"], NN[split].get(q, []) if mode != "none" else [],
                                  chunk_hits=hits.get(split, {}).get(q), answer=row["final_answer"], noise=noise,
                                  rng=r, pool_qids=pool_qids, tok=tok))
    return out

os.makedirs(WORK + "/data", exist_ok=True)
for mode in RUN:
    for name, items, split, noise in (("train", sub, "train", NOISE if mode != "none" else 0.0), ("dev", dev_rows, "dev", 0.0)):
        ex = build(mode, items, split, noise)
        with open("%s/data/%s_%s.jsonl" % (WORK, name, mode), "w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\\n")
    lens = [len(tok(e["system"] + e["user"], add_special_tokens=False)["input_ids"]) for e in ex]
    print("mode %-5s | prompt tokens: mean %.0f, max %d (limit leaves %d for the answer)" % (mode, sum(lens) / len(lens), max(lens), MAX_LEN - max(lens)))
ex = build(RUN[0], dev_rows[:1], "dev", 0.0)[0]
print("\\n--- example prompt (mode %s) ---\\n%s\\n--- target answer starts: %s" % (RUN[0], ex["user"][:1800], ex["answer"][:200].replace("\\n", " ")))
'''),
        md("## 2. Download the model once (so the two parallel runs do not race on the download)"),
        code('''
if not EXTRA:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
import torch
N_GPU = torch.cuda.device_count()
print("GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
'''),
        md("## 3. Pilot - 3 optimizer steps to measure speed and memory before the long run\nIf this fails with an out-of-memory error, lower `MAX_LEN` (e.g. 2048). If the loss is `nan`, switch `MODEL` to `Qwen/Qwen2.5-1.5B-Instruct`."),
        code('''
def cmd(mode, out, extra=()):
    return [sys.executable, "train_qlora.py", "--train", "%s/data/train_%s.jsonl" % (WORK, mode), "--dev", "%s/data/dev_%s.jsonl" % (WORK, mode),
            "--out", out, "--model", MODEL, "--max_len", str(MAX_LEN), "--grad_accum", str(GRAD_ACCUM),
            "--epochs", str(EPOCHS), "--train_hours", str(TRAIN_HOURS)] + list(extra) + EXTRA

mode = RUN[0]
r = subprocess.run(cmd(mode, WORK + "/pilot", ["--max_steps", "3", "--grad_accum", "4", "--save_steps", "1000", "--eval_steps", "1000"]),
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false"), capture_output=True, text=True)
print(r.stdout[-1800:]); print(r.stderr[-1500:] if r.returncode else "")
assert r.returncode == 0, "pilot failed - read the error above"
s = json.load(open(WORK + "/pilot/train_summary.json"))
per_example = s["sec_per_step"] / 4
n = len(open("%s/data/train_%s.jsonl" % (WORK, mode)).readlines())
proj = per_example * n * EPOCHS / 3600
print("pilot: %.1f s per example (~%d tokens/s) -> %d examples x %.1f epochs ~ %.1f h per run" % (per_example, s["tokens_per_sec_est"], n, EPOCHS, proj))
if proj > TRAIN_HOURS:
    print("WARNING: projected time exceeds TRAIN_HOURS=%.1f - the run will stop early. Reduce N_TRAIN or EPOCHS." % TRAIN_HOURS)
import shutil; shutil.rmtree(WORK + "/pilot", ignore_errors=True)
'''),
        md("## 4. Train - one process per GPU (waves of N_GPU runs)"),
        code('''
os.makedirs(WORK + "/logs", exist_ok=True)
slots = max(1, N_GPU)
for w in range(0, len(RUN), slots):
    wave = RUN[w:w + slots]
    procs = []
    for gpu, mode in enumerate(wave):
        log = open("%s/logs/%s.log" % (WORK, mode), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        procs.append((mode, subprocess.Popen(cmd(mode, "%s/adapters/%s" % (WORK, mode)), env=env, stdout=log, stderr=subprocess.STDOUT)))
        print("started", mode, "on GPU", gpu)
    t0 = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(120)
        for mode, p in procs:
            lines = [l for l in open("%s/logs/%s.log" % (WORK, mode), errors="ignore").read().splitlines()
                     if any(k in l for k in ("loss", "DONE", "Error", "error", "Traceback", "examples", "trainable"))]
            print("[%4.0f min] %-5s %s" % ((time.time() - t0) / 60, mode, lines[-1][:170] if lines else "(no output yet)"), flush=True)
    for mode, p in procs:
        print("finished", mode, "exit code", p.returncode)
'''),
        md("## 5. Results - training and dev loss, and whether each run finished cleanly"),
        code('''
for mode in RUN:
    f = "%s/adapters/%s/train_summary.json" % (WORK, mode)
    if not os.path.exists(f):
        print(mode, ": NO SUMMARY - see logs/%s.log" % mode); continue
    s = json.load(open(f))
    print("%-5s status=%s steps=%d (%.1f h) | train loss %s -> %s | dev loss %s" % (mode, s["status"], s["steps"], s["elapsed_s"] / 3600,
          s["loss_log"][0][1] if s["loss_log"] else "-", s["loss_log"][-1][1] if s["loss_log"] else "-", s["eval_log"]))
'''),
        md("## 6. Sanity check - does the adapter write a sensible answer? (3 dev questions per mode)"),
        code('''
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
if EXTRA:
    print("skipped in CPU smoke test")
else:
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    for mode in RUN:
        base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, torch_dtype=torch.float16, device_map={"": 0})
        model = PeftModel.from_pretrained(base, "%s/adapters/%s/final" % (WORK, mode)).eval()
        for e in [json.loads(l) for l in open("%s/data/dev_%s.jsonl" % (WORK, mode))][:3]:
            text = tok.apply_chat_template([{"role": "system", "content": e["system"]}, {"role": "user", "content": e["user"]}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=350, do_sample=False, repetition_penalty=1.05)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            print("=" * 80, "\\n[%s] Q: %s" % (mode, e["user"].split("Question: ")[-1][:150]), "\\n", gen[:700], "\\n... (%d words generated)" % len(gen.split()))
        del model, base; gc.collect(); torch.cuda.empty_cache()
'''),
        md("""
## After it finishes
Output tab -> **New Dataset** (e.g. `nlp-qna-adapters`): it contains `adapters/<mode>/final` (the LoRA weights), the training summaries and logs.
Paste me the **results table (section 5)** and the **sample answers (section 6)**. Phase 5 then generates answers for the 1,500 hold-out rows
and the 1,000 unseen questions and scores every system with the shared evaluation harness.
"""),
    ]
    return notebook(cells)


def build_phase5_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 05 - Phase 5: inference, evaluation and the final predictions

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
> This is the **inference notebook**: it loads the trained LoRA adapters and needs no training.

**Inputs (Add Input):**
1. the Phase 0 dataset(s) holding `train_processed.csv` **and** `test_processed.csv` (the unseen 1,000 questions),
2. the Phase 2 dataset `nlp-qna-retrieval` (`question_neighbours.json`),
3. the adapters dataset `nlp-qna-adapters` (`nn/` and `none/`).

**What it does**
1. **Exact-match router** - an unseen question whose text matches a training question exactly gets that question's stored (medoid) answer. No model call.
2. Every other unseen question is answered by the fine-tuned model with its 2 nearest training Q&A pairs in the prompt.
3. **Hold-out evaluation** on questions the model never saw (a grouped split, so no question text crosses train/test): four systems on the same questions -
   B0a nearest-neighbour answer, **B1** zero-shot base model, **B2** fine-tuned without retrieval, **proposed** fine-tuned with retrieval - plus a bootstrap confidence interval.
4. The router is scored separately, because it only applies to repeated questions. The combined number for the unseen set is printed as a clearly labelled *estimate*.
5. Writes `xt_predictions.csv` and `metrics.json`.

Generation runs one worker per GPU and **appends every finished batch to disk**: if the session dies, re-run and it resumes.
"""),
        code("!pip install -q -U peft bitsandbytes accelerate huggingface_hub rouge-score sacrebleu"),
        code("%%writefile prompts.py\n" + src("prompts.py")),
        code("%%writefile eval_harness.py\n" + src("eval_harness.py")),
        code("%%writefile infer.py\n" + src("infer.py")),
        md("## 0. Configuration"),
        code('''
import glob, os, sys, json, gzip, csv, random, time, subprocess, collections as C
csv.field_size_limit(10 ** 9)
_missing = [f for f in ("prompts.py", "eval_harness.py", "infer.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - run the %%writefile cells at the top first (use Run All, do not start from a lower cell)"
sys.path.insert(0, os.getcwd())

import torch
if not os.environ.get("P5_CPU"):   # (CPU smoke-test hook, never set on Kaggle)
    assert torch.cuda.is_available(), ("No GPU attached to this session. Right panel -> Session options -> Accelerator -> "
                                       "GPU T4 x2 (the session restarts), then Run All from the top.")
    print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

MODEL = os.environ.get("P5_MODEL", "Qwen/Qwen2.5-3B-Instruct")
N_HOLDOUT = 300            # hold-out questions scored (same ones for every system). 1500 = the whole hold-out set (about 5x slower)
UNSEEN_ADAPTER = "nn"      # which adapter answers the non-repeated unseen questions: "nn" (with retrieval) or "none"
MAX_NEW = 768              # generation limit in tokens (the median reference answer is about 417 words ~ 600 tokens)
BATCH = 12
GEN_HOURS = 9.0            # each worker stops cleanly after this long; re-run to resume
BERTSCORE = False          # True adds BERTScore (downloads roberta-large; slower)
WORK = "/kaggle/working"

def find(pattern, required=True):
    hit = glob.glob("/kaggle/input/**/" + pattern, recursive=True)
    if not hit and required:
        print("not found:", pattern, "- files under /kaggle/input (first 40):")
        for i, (dp, dn, fn) in enumerate(os.walk("/kaggle/input")):
            for x in fn[:8]:
                print("  ", os.path.join(dp, x))
            if i > 40:
                break
        raise AssertionError("attach a dataset containing " + pattern)
    return hit[0] if hit else None

def op(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")

TRAIN_CSV, UNSEEN_CSV, NN_PATH = find("train_processed*.csv"), find("test_processed*.csv"), find("question_neighbours.json*")
AD_DIR = os.path.dirname(os.path.dirname(find("nn/adapter_config.json")))
assert os.path.exists(AD_DIR + "/none/adapter_config.json"), "the adapters dataset must contain nn/ and none/ folders"
print("adapters:", AD_DIR, "| model:", MODEL, "| hold-out questions:", N_HOLDOUT, "| unseen adapter:", UNSEEN_ADAPTER)
'''),
        md("## 1. Router and prompts (CPU)"),
        code('''
import prompts as P
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
unseen = list(csv.DictReader(open(UNSEEN_CSV, encoding="utf-8")))
NN = json.load(op(NN_PATH))
ctx = P.Context(rows)
qid2row = ctx.qid2row

# exact-match router: unseen question text == a training question -> that group's medoid answer
medoid = {r["group_id"]: r for r in rows if r["is_medoid"] == "1"}
routed = {u["question_id"]: medoid[u["matched_group_id"]]["final_answer"]
          for u in unseen if u["in_train"] == "1" and u["matched_group_id"] in medoid}
gen_unseen = [u for u in unseen if u["question_id"] not in routed]
print("unseen questions: %d | routed by exact match: %d (%.1f%%) | to generate: %d" % (len(unseen), len(routed), 100 * len(routed) / len(unseen), len(gen_unseen)))

test_rows = [r for r in rows if r["split"] == "test"]
hold = random.Random(42).sample(test_rows, min(N_HOLDOUT, len(test_rows))) if N_HOLDOUT else test_rows
print("hold-out rows (never seen in training):", len(hold), "of", len(test_rows))

def mk(adapter, prio, tag, qid, question, mode, split):
    ex = P.make_example(ctx, mode, qid, question, NN[split].get(qid, []) if mode != "none" else [], tok=tok)
    return {"uid": "%s|%s|%s" % (adapter, tag, qid), "adapter": adapter, "prio": prio, "system": ex["system"], "user": ex["user"]}

jobs = []
for r in hold:
    q = r["question_id"]
    jobs.append(mk("nn", 0, "hold", q, r["question"], "nn", "test"))
    jobs.append(mk("none", 0, "hold", q, r["question"], "none", "test"))
    jobs.append(mk("base", 0, "hold", q, r["question"], "none", "test"))   # B1: same prompt as B2, adapter switched off
for u in gen_unseen:
    jobs.append(mk(UNSEEN_ADAPTER, 1, "xt", u["question_id"], u["question"], "nn" if UNSEEN_ADAPTER == "nn" else "none", "xt"))
os.makedirs(WORK + "/gen", exist_ok=True)
with open(WORK + "/jobs.jsonl", "w", encoding="utf-8") as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + "\\n")
lens = [len(tok(j["system"] + j["user"], add_special_tokens=False)["input_ids"]) for j in jobs]
print("jobs:", len(jobs), dict(C.Counter(j["adapter"] for j in jobs)), "| prompt tokens mean %.0f max %d" % (sum(lens) / len(lens), max(lens)))
'''),
        md("## 2. Generate - one worker per GPU (resumable)\nThe first batches are the longest prompts, so a memory problem shows up immediately (the worker halves its batch size by itself)."),
        code('''
if not os.environ.get("P5_CPU"):
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
N_GPU = max(1, torch.cuda.device_count())
workers = [["nn"], ["none", "base"]] if N_GPU >= 2 else [["nn", "none", "base"]]
procs = []
for gpu, names in enumerate(workers):
    log = open("%s/gen/worker%d.log" % (WORK, gpu), "a")
    cmd = [sys.executable, "infer.py", "--jobs", WORK + "/jobs.jsonl", "--adapter_dir", AD_DIR, "--adapters", ",".join(names),
           "--out", "%s/gen/gen_gpu%d.jsonl" % (WORK, gpu), "--model", MODEL, "--max_new", str(MAX_NEW), "--batch", str(BATCH), "--hours", str(GEN_HOURS)]
    if os.environ.get("P5_CPU"):
        cmd.append("--no_quant")
    procs.append((gpu, subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false"), stdout=log, stderr=subprocess.STDOUT)))
    print("started worker", gpu, "adapters", names)
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(120)
    for gpu, p in procs:
        lines = open("%s/gen/worker%d.log" % (WORK, gpu), errors="ignore").read().splitlines()
        print("[%4.0f min] worker %d: %s" % ((time.time() - t0) / 60, gpu, lines[-1][:150] if lines else "(no output yet)"), flush=True)
for gpu, p in procs:
    print("worker", gpu, "exit code", p.returncode)
    if p.returncode:
        print(open("%s/gen/worker%d.log" % (WORK, gpu), errors="ignore").read()[-2000:])
'''),
        md("## 3. Collect the answers"),
        code('''
gen = {}
for f in glob.glob(WORK + "/gen/gen_gpu*.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            gen[d["uid"]] = d
missing = [j["uid"] for j in jobs if j["uid"] not in gen]
print("generated: %d of %d jobs | missing: %d" % (len(gen), len(jobs), len(missing)))
if missing:
    print("WARNING: some jobs are missing (a worker stopped early). Run this notebook again - it resumes. First missing:", missing[:3])
cut = sum(d["hit_limit"] for d in gen.values())
print("answers cut off at MAX_NEW=%d tokens: %d (%.1f%%)" % (MAX_NEW, cut, 100 * cut / max(1, len(gen))))
'''),
        md("## 4. Hold-out evaluation - the main results table for the report"),
        code('''
import numpy as np, pandas as pd
import eval_harness as H
def nearest(split, q):
    nb = NN[split].get(q)
    return qid2row[nb[0][0]]["final_answer"] if nb else ""

ids = [r["question_id"] for r in hold if all(("%s|hold|%s" % (a, r["question_id"])) in gen for a in ("nn", "none", "base"))]
refs = {r["question_id"]: r["final_answer"] for r in hold if r["question_id"] in set(ids)}
systems = {"B0a nearest-neighbour answer (no model)": {q: nearest("test", q) for q in ids}}
empty = {}
for name, ad in (("B1 zero-shot base model", "base"), ("B2 fine-tuned, no retrieval", "none"), ("PROPOSED fine-tuned + retrieval", "nn")):
    p, e = {}, 0
    for q in ids:
        t = gen["%s|hold|%s" % (ad, q)]["text"]
        if not t.strip():
            t, e = nearest("test", q), e + 1   # an empty generation falls back to the nearest answer (counted)
        p[q] = t
    systems[name] = p
    empty[name] = e
print("scored questions:", len(ids), "| empty generations replaced by the fallback:", empty)
res = {n: H.evaluate(p, refs, bertscore=BERTSCORE) for n, p in systems.items()}
cols = ["rouge1", "rouge2", "rougeL", "bleu", "len_ratio_mean", "fact_f1"] + (["bertscore_f1"] if BERTSCORE else [])
table = pd.DataFrame({n: {c: r[c] for c in cols} for n, r in res.items()}).T
pd.set_option("display.width", 200)
print(table.round(4).to_string())

# paired bootstrap on ROUGE-L (resample questions, 2000 times)
from rouge_score import rouge_scorer
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
per = {n: np.array([sc.score(refs[q], p[q])["rougeL"].fmeasure for q in ids]) for n, p in systems.items()}
idx = np.random.default_rng(0).integers(0, len(ids), (2000, len(ids)))
def diff_ci(a, b):
    d = (per[a] - per[b])[idx].mean(axis=1)
    return "%+.4f  [95%% CI %+.4f, %+.4f]" % (per[a].mean() - per[b].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
names = list(systems)
print("\\nROUGE-L difference (paired bootstrap):")
print("  proposed - B2 (value of retrieval):", diff_ci(names[3], names[2]))
print("  proposed - B1 (value of fine-tuning):", diff_ci(names[3], names[1]))
print("  proposed - B0a (value of generating):", diff_ci(names[3], names[0]))
'''),
        md("## 5. The router, scored separately\nOnly repeated questions can be routed, so this is measured on training questions that appear more than once: the stored answer is compared with a *different* human answer to the same question."),
        code('''
groups = C.defaultdict(list)
for r in rows:
    groups[r["group_id"]].append(r)
rep = [g for g in groups.values() if len(g) >= 2]
rr = random.Random(7)
held, p_med, p_rand = {}, {}, {}
for g in rr.sample(rep, min(1500, len(rep))):
    med = next((x for x in g if x["is_medoid"] == "1"), g[0])
    h = rr.choice([x for x in g if x is not med])            # a human answer that the stored one is compared with
    held[h["question_id"]] = h["final_answer"]
    p_med[h["question_id"]] = med["final_answer"]            # the router's answer: the stored medoid
    p_rand[h["question_id"]] = rr.choice([x for x in g if x is not h])["final_answer"]
router_res = H.evaluate(p_med, held)
rand_res = H.evaluate(p_rand, held)
print("repeated-question groups: %d (of %d groups) | scored: %d" % (len(rep), len(groups), len(held)))
print("router (stored medoid answer)  : rougeL %.4f  rouge1 %.4f  bleu %.4f  fact_f1 %.4f" % tuple(router_res[k] for k in ("rougeL", "rouge1", "bleu", "fact_f1")))
print("random other human answer      : rougeL %.4f  rouge1 %.4f  bleu %.4f  fact_f1 %.4f" % tuple(rand_res[k] for k in ("rougeL", "rouge1", "bleu", "fact_f1")))
share = len(routed) / len(unseen)
prop = res[names[3]]
print("\\nESTIMATE for the unseen set (not a measurement): %.1f%% routed x router score + %.1f%% generated x hold-out score of the proposed model" % (100 * share, 100 * (1 - share)))
for k in ("rougeL", "rouge1", "fact_f1"):
    print("   %-8s %.4f" % (k, share * router_res[k] + (1 - share) * prop[k]))
'''),
        md("## 6. Write the predictions for the unseen set"),
        code('''
final, source, fallback = {}, {}, 0
for u in unseen:
    q = u["question_id"]
    if q in routed:
        final[q], source[q] = routed[q], "router"
        continue
    d = gen.get("%s|xt|%s" % (UNSEEN_ADAPTER, q))
    if d and d["text"].strip():
        final[q], source[q] = d["text"], "generated"
    else:
        final[q], source[q] = nearest("xt", q), "fallback_nearest"
        fallback += 1
assert len(final) == len(unseen) == 1000 and all(v.strip() for v in final.values()), "predictions incomplete"
with open(WORK + "/xt_predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question_id", "question", "final_answer"])
    for u in unseen:
        w.writerow([u["question_id"], u["question"], final[u["question_id"]]])
with open(WORK + "/xt_predictions_detailed.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question_id", "source", "final_answer"])
    for u in unseen:
        w.writerow([u["question_id"], source[u["question_id"]], final[u["question_id"]]])
print("wrote xt_predictions.csv:", len(final), "rows |", dict(C.Counter(source.values())))
if fallback:
    print("WARNING: %d unseen questions used the nearest-answer fallback (missing generations) - re-run to resume." % fallback)
json.dump({"holdout_n": len(ids), "systems": res, "router": router_res, "router_random_other": rand_res,
           "routed_share": share, "bootstrap_rougeL": {"proposed_minus_B2": diff_ci(names[3], names[2]), "proposed_minus_B1": diff_ci(names[3], names[1]),
           "proposed_minus_B0a": diff_ci(names[3], names[0])}, "config": {"N_HOLDOUT": N_HOLDOUT, "MAX_NEW": MAX_NEW, "UNSEEN_ADAPTER": UNSEEN_ADAPTER, "MODEL": MODEL}},
          open(WORK + "/metrics.json", "w"), indent=1, default=float)
'''),
        md("## 7. Look at a few answers (for the report's qualitative section)"),
        code('''
for q in ids[:3]:
    print("=" * 90, "\\nQ:", qid2row[q]["question"][:200])
    print("--- reference:", refs[q][:450].replace("\\n", " "))
    print("--- proposed :", systems[names[3]][q][:450].replace("\\n", " "))
    print("--- B2       :", systems[names[2]][q][:300].replace("\\n", " "))
'''),
        md("""
## After it finishes
Output tab: `xt_predictions.csv` (the deliverable), `xt_predictions_detailed.csv` (marks each answer as router / generated), `metrics.json`, `gen/` (raw generations).
Send me the **tables printed in sections 4 and 5**.
"""),
    ]
    return notebook(cells)


SWEEP_PAYLOAD = "C:/nlp_data/diag/sweep_payload.json"


def build_decode_sweep_nb():
    import base64
    import gzip as _gz

    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    blob = base64.b64encode(_gz.compress(open(SWEEP_PAYLOAD, "rb").read(), 9)).decode("ascii")
    lines = "\n".join(blob[i:i + 120] for i in range(0, len(blob), 120))
    cells = [
        md("""
# 06 - Decoding sweep: can the repetition loops be fixed without retraining?

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
> **Input (Add Input):** only the adapters dataset `nlp-qna-adapters` (the `nn/` adapter). The 50 test questions are embedded below.

**Why.** In Phase 5, 130 of 300 hold-out answers hit the 768-token cap, and most of those were repetition loops that start early
(median at word ~130), so the cap is not the cause. Greedy decoding with a near-zero repetition penalty is a known cause of loops.

**What.** The same trained adapter (`nn`) and the **exact Phase 5 prompts** for the first 50 of the 300 Phase 5 hold-out questions are answered
under 7 decoding settings (all with a 1,280-token cap):

| config | decoding |
|---|---|
| A | greedy, HF repetition penalty 1.05 (= Phase 5 decoding, only the cap is higher) |
| B | greedy, HF repetition penalty 1.15 (HF's penalty also hits words in the prompt) |
| C | greedy, penalty 1.15 on the answer's own tokens only |
| D | greedy, penalty 1.05 + no repeated 12-token n-gram inside the answer |
| E | greedy, answer-only penalty 1.1 + no repeated 12-token n-gram |
| F | sampling (temperature 0.7, top-p 0.9), penalty 1.05 |
| G | sampling + answer-only penalty 1.1 + no repeated 12-token n-gram |

Scored with the same harness as Phase 5 against the human reference answers, next to B0a (copy the nearest answer) and the saved
Phase 5 answers for the same 50 questions. **N = 50, so differences under ~0.02 ROUGE-L are noise; the table chooses a setting, it is not a final result.**
"""),
        code("!pip install -q -U peft bitsandbytes accelerate huggingface_hub rouge-score sacrebleu"),
        code("%%writefile infer.py\n" + src("infer.py")),
        code("%%writefile decode_sweep.py\n" + src("decode_sweep.py")),
        code("%%writefile eval_harness.py\n" + src("eval_harness.py")),
        md("## 0. The 50 test questions (embedded: prompts, references, B0a and Phase 5 answers)"),
        code('PAYLOAD_B64 = """\n' + lines + '\n"""\n'
             'import base64, gzip, json\n'
             'payload = json.loads(gzip.decompress(base64.b64decode("".join(PAYLOAD_B64.split()))).decode("utf-8"))\n'
             'json.dump(payload, open("payload.json", "w", encoding="utf-8"), ensure_ascii=False)\n'
             'print(payload["source"], "| items:", len(payload["items"]))'),
        md("## 1. Configuration"),
        code('''
import glob, os, sys, json, time, subprocess, collections as C
_missing = [f for f in ("infer.py", "decode_sweep.py", "eval_harness.py", "payload.json") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - use Run All from the top"
sys.path.insert(0, os.getcwd())
import torch
assert torch.cuda.is_available(), "No GPU. Right panel -> Session options -> Accelerator -> GPU T4 x2, then Run All."
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

MODEL = "Qwen/Qwen2.5-3B-Instruct"
BATCH = 10
WORK = "/kaggle/working"
hit = glob.glob("/kaggle/input/**/nn/adapter_config.json", recursive=True)
assert hit, "attach the adapters dataset (nlp-qna-adapters) with the nn/ folder"
AD_DIR = os.path.dirname(os.path.dirname(hit[0]))
import decode_sweep as DS
ALL = list(DS.CONFIGS)
if torch.cuda.device_count() >= 2:
    workers = [[c for c in ALL if c[0] in "ACEG"], [c for c in ALL if c[0] in "BDF"]]
else:
    workers = [ALL]
print("adapters:", AD_DIR)
for i, w in enumerate(workers):
    print("GPU", i, "->", w)
'''),
        md("## 2. Generate (one worker per GPU, resumable)"),
        code('''
from huggingface_hub import snapshot_download
snapshot_download(MODEL)
os.makedirs(WORK + "/sweep", exist_ok=True)
procs = []
for gpu, names in enumerate(workers):
    log = open("%s/sweep/worker%d.log" % (WORK, gpu), "a")
    cmd = [sys.executable, "decode_sweep.py", "--payload", "payload.json", "--configs", ",".join(names), "--adapter_dir", AD_DIR,
           "--out", "%s/sweep/gen_gpu%d.jsonl" % (WORK, gpu), "--model", MODEL, "--batch", str(BATCH)]
    procs.append((gpu, subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false"),
                                        stdout=log, stderr=subprocess.STDOUT)))
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(90)
    for gpu, p in procs:
        lines = open("%s/sweep/worker%d.log" % (WORK, gpu), errors="ignore").read().splitlines()
        print("[%4.0f min] worker %d: %s" % ((time.time() - t0) / 60, gpu, lines[-1][:150] if lines else "(starting)"), flush=True)
for gpu, p in procs:
    print("worker", gpu, "exit code", p.returncode)
    if p.returncode:
        print(open("%s/sweep/worker%d.log" % (WORK, gpu), errors="ignore").read()[-3000:])
'''),
        md("## 3. Score every configuration on the same 50 questions"),
        code('''
import re
import numpy as np, pandas as pd
import eval_harness as H
from rouge_score import rouge_scorer

items = {it["qid"]: it for it in payload["items"]}
ids = list(items)
refs = {q: items[q]["reference"] for q in ids}
gen = {}
for f in glob.glob(WORK + "/sweep/gen_gpu*.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); gen[d["uid"]] = d

WORD = re.compile(r"\\S+")
def is_loop(text, n=8, min_rep=3):
    w = [x.lower().strip(".,;:*#-()") for x in WORD.findall(text)]
    c = C.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return bool(c) and max(c.values()) >= min_rep

systems = {"B0a copy nearest answer": {q: items[q]["b0a"] for q in ids},
           "Phase5 saved (greedy rp1.05, cap 768)": {q: items[q]["phase5_text"] for q in ids}}
extra = {"B0a copy nearest answer": {}, "Phase5 saved (greedy rp1.05, cap 768)": {"hit_cap": sum(items[q]["phase5_hit_limit"] for q in ids)}}
for c in ALL:
    got = [gen.get("%s|%s" % (c, q)) for q in ids]
    if not all(got):
        print("WARNING: config %s incomplete (%d/%d) - re-run to resume" % (c, sum(map(bool, got)), len(ids)))
        continue
    systems[c] = {q: (g["text"] if g["text"].strip() else items[q]["b0a"]) for q, g in zip(ids, got)}
    extra[c] = {"hit_cap": sum(g["hit_limit"] for g in got), "sec_per_answer": float(np.mean([g["sec_per_answer"] for g in got])),
                "empty": sum(not g["text"].strip() for g in got)}

res = {n: H.evaluate(p, refs) for n, p in systems.items()}
for n, p in systems.items():
    res[n].update(extra[n])
    res[n]["loops_8gram_x3"] = sum(is_loop(p[q]) for q in ids)
    res[n]["median_words"] = float(np.median([len(p[q].split()) for q in ids]))
cols = ["rougeL", "rouge1", "rouge2", "bleu", "fact_f1", "len_ratio_mean", "median_words", "hit_cap", "loops_8gram_x3", "empty", "sec_per_answer"]
table = pd.DataFrame({n: {c: r.get(c, np.nan) for c in cols} for n, r in res.items()}).T
pd.set_option("display.width", 250)
print("N = %d hold-out questions (reference median %d words)" % (len(ids), np.median([len(refs[q].split()) for q in ids])))
print(table.round(4).to_string())

sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
per = {n: np.array([sc.score(refs[q], p[q])["rougeL"].fmeasure for q in ids]) for n, p in systems.items()}
idx = np.random.default_rng(0).integers(0, len(ids), (2000, len(ids)))
def diff_ci(a, b):
    d = (per[a] - per[b])[idx].mean(axis=1)
    return "%+.4f  [95%% CI %+.4f, %+.4f]" % (per[a].mean() - per[b].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
P5N, B0N = "Phase5 saved (greedy rp1.05, cap 768)", "B0a copy nearest answer"
boot = {}
print("\\nROUGE-L paired bootstrap (N=%d):" % len(ids))
for n in systems:
    if n not in (P5N, B0N):
        boot[n] = {"minus_phase5": diff_ci(n, P5N), "minus_B0a": diff_ci(n, B0N)}
        print("  %-30s vs Phase5 %s | vs B0a %s" % (n, boot[n]["minus_phase5"], boot[n]["minus_B0a"]))
json.dump({"n": len(ids), "configs": {c: DS.settings(c) for c in ALL}, "results": res, "bootstrap_rougeL": boot},
          open(WORK + "/sweep_results.json", "w"), indent=1, default=float)
print("\\nwrote sweep_results.json")
'''),
        md("## 4. Look at the question where Phase 5 looped worst, under each setting"),
        code('''
def rep_score(t):
    w = t.lower().split(); c = C.Counter(tuple(w[i:i + 8]) for i in range(len(w) - 7))
    return max(c.values()) if c else 0
q = max(ids, key=lambda q: rep_score(items[q]["phase5_text"]))
print("Q:", items[q]["question"])
for n, p in systems.items():
    t = p[q].replace("\\n", " ")
    print("=" * 100, "\\n%s (%d words, rougeL %.3f)\\n  ...%s" % (n, len(t.split()), per[n][ids.index(q)], t[-500:]))
'''),
        md("""
## After it finishes
Send me the **table from section 3** and the **bootstrap lines** (or just tell me the notebook name; I can download the output).
Output tab: `sweep_results.json`, `sweep/gen_gpu*.jsonl` (every answer).
"""),
    ]
    return notebook(cells)


def build_rerank_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 07 - Neighbour re-ranker: give the model better reference answers

> Run with **Save Version -> Save & Run All (Commit)**. **Accelerator: GPU T4 x1 recommended** (the cross-encoder scores
> 160k question/answer pairs in minutes); **CPU works too**, but then the question-vs-answer cross-encoder feature is switched off. Internet = On.

**Inputs (Add Input):** `train_processed.csv` (dataset `train-processed1`), `test_processed.csv` (dataset `train-processed`),
`question_neighbours.json` (dataset `nlp-qna-retrieval`).

**Why.** The generated answer's quality follows the quality of the reference answers in its prompt (correlation 0.63 on the 300 hold-out rows),
but the current retrieval order puts the best of the 10 neighbours first only 23% of the time (best-of-10 would score 0.30 ROUGE-L vs 0.215 for top-1).

**What.** For every query and each of its 10 retrieved training questions, the label is the ROUGE-L between that neighbour's answer and the query's
true answer. A gradient-boosted model learns to predict it from retrieval scores, idf-weighted question overlap, answer lengths, consensus among the
10 answers, and cross-encoder scores. Model choice on **dev**; the reported number is on the **1,500 hold-out rows**. Training-split queries are
re-ranked by **5-fold cross-fitting by question group** (no prompt is chosen by a model that saw its own answer).

**Output:** `question_neighbours_reranked.json.gz` (same format as `question_neighbours.json`, drop-in for training/inference) and `rerank_report.json`.
"""),
        code("!pip install -q rouge-score"),
        code("%%writefile rerank.py\n" + src("rerank.py")),
        md("## 1. Inputs and settings"),
        code('''
import glob, os, sys, json
sys.path.insert(0, os.getcwd())
assert os.path.exists("rerank.py"), "run the %%writefile cell first (use Run All)"
def find(pattern):
    hit = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    assert hit, "attach a dataset containing " + pattern
    return hit[0]
TRAIN_CSV, UNSEEN_CSV = find("train_processed*.csv"), find("test_processed*.csv")
NN_PATH = find("question_neighbours.json*")
import torch
GPU = torch.cuda.is_available()
CE_Q = True          # cross-encoder: query question vs neighbour question (cheap)
CE_A = GPU           # cross-encoder: query question vs neighbour answer (needs a GPU to be quick)
OUT = "/kaggle/working"
print("train:", TRAIN_CSV, "| unseen:", UNSEEN_CSV, "| neighbours:", NN_PATH, "| GPU:", GPU, "| CE_Q", CE_Q, "| CE_A", CE_A)
'''),
        md("## 2. Labels, features, models, evaluation, re-ranked lists (about 15-25 min)"),
        code('''
import rerank
args = ["--train_csv", TRAIN_CSV, "--unseen_csv", UNSEEN_CSV, "--nn", NN_PATH, "--out", OUT]
if CE_Q: args.append("--ce_q")
if CE_A: args.append("--ce_a")
report = rerank.main(args)
'''),
        md("""
## After it finishes
Send me the notebook name (I download the output with the Kaggle CLI), or paste the **dev / test tables** and the **"chosen on dev"** line.
Output tab: `question_neighbours_reranked.json.gz`, `rerank_report.json`.
"""),
    ]
    return notebook(cells)


def build_retrain_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 08 - Retrain the answer model: all 12,825 training rows, re-ranked references, both GPUs

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
> One commit run is limited to 12 h. Training stops cleanly after `TRAIN_HOURS` and saves a checkpoint; if it has not finished,
> run the notebook **again** with this notebook's own output attached (see the end) and it **resumes** from that checkpoint.

**Inputs (Add Input):**
1. `train_processed.csv` (dataset `train-processed1`)
2. `question_neighbours_reranked.json.gz` (Your Work -> the re-ranker notebook 07 output)
3. *second session only:* this notebook's previous output (Your Work -> this notebook)

**What changed vs Phase 4 (and why)**

| | Phase 4 | this run | evidence |
|---|---|---|---|
| training rows | 4,000 random, 1 epoch | **all 12,825**, 1 epoch | loss still ~1.45; fine-tuning on little data doubled repetition loops |
| reference answers in the prompt | retrieval top-2 | **re-ranked top-2** (cross-fitted for train rows) | generated quality follows reference quality (r = 0.63); re-ranking +0.015 top-1 ROUGE-L on 1,500 hold-out rows |
| max length | 2,560 | **3,072** | 266/4,000 targets were cut and never showed an ending |
| GPUs | one adapter per GPU | **both GPUs, one model (DDP)** | 802 steps x ~83 s on one T4 = ~18 h |

Unchanged: Qwen2.5-3B-Instruct, 4-bit NF4, LoRA r=16/alpha=32 on all linear layers, lr 2e-4 cosine, effective batch 16,
loss on answer tokens only, 20% RAFT-style distractor references, the same system prompt.
"""),
        code("!pip install -q -U peft bitsandbytes accelerate huggingface_hub"),
        code("%%writefile prompts.py\n" + src("prompts.py")),
        code("%%writefile train_qlora.py\n" + src("train_qlora.py")),
        code("%%writefile infer.py\n" + src("infer.py")),
        md("## 0. Configuration and resume detection"),
        code('''
import glob, os, sys, json, gzip, csv, random, time, shutil, subprocess, collections as C
csv.field_size_limit(10 ** 9)
_missing = [f for f in ("prompts.py", "train_qlora.py", "infer.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - use Run All from the top"
sys.path.insert(0, os.getcwd())
import torch
EXTRA = os.environ.get("P8_EXTRA", "").split()        # CPU smoke-test hook only; empty on Kaggle
if not EXTRA:
    assert torch.cuda.is_available(), "No GPU. Session options -> Accelerator -> GPU T4 x2, then Run All."
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else int(os.environ.get("P8_NPROC", "1"))
print("GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

MODEL = os.environ.get("P8_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODE = "nn"
N_TRAIN = int(os.environ.get("P8_NTRAIN", "0"))   # 0 = all 12,825 training rows
EPOCHS = 1.0
MAX_LEN = 3072
EFFECTIVE_BATCH = 16
NOISE = 0.2
N_DEV = 60
TRAIN_HOURS = float(os.environ.get("P8_HOURS", "10.5"))   # leaves ~1.5 h of the 12 h session for setup and the final cells
SAVE_STEPS = 25                                          # ~20 min of training between checkpoints
WORK = os.environ.get("P8_WORK", "/kaggle/working")
INPUT = os.environ.get("P8_INPUT", "/kaggle/input")
OUT = WORK + "/adapters/" + MODE

def find(pattern, required=True):
    hit = sorted(glob.glob(INPUT + "/**/" + pattern, recursive=True))
    if not hit and required:
        raise AssertionError("attach a dataset/notebook output containing " + pattern)
    return hit[0] if hit else None

def op(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")

TRAIN_CSV = find("train_processed*.csv")
NN_PATH = find("question_neighbours_reranked.json*")
os.makedirs(WORK + "/data", exist_ok=True)

# resume: the latest checkpoint and the exact training prompts of a previous session of this notebook
import re
prev = [p.replace("\\\\", "/") for p in glob.glob(INPUT + "/**/adapters/%s/checkpoint-*/trainer_state.json" % MODE, recursive=True)]
prev.sort(key=lambda p: int(re.search(r"checkpoint-(\\d+)", p).group(1)))
RESUMED_FROM = None
if prev and not os.path.exists(OUT):
    ck = os.path.dirname(prev[-1])
    RESUMED_FROM = os.path.basename(ck)
    shutil.copytree(ck, os.path.join(OUT, RESUMED_FROM))
    root = ck.rsplit("/adapters/", 1)[0]
    for f in ("train_%s.jsonl" % MODE, "dev_%s.jsonl" % MODE):
        p = os.path.join(root, "data", f)
        assert os.path.exists(p), "previous output has a checkpoint but no data/%s - cannot resume safely" % f
        shutil.copy(p, os.path.join(WORK, "data", f))
    for d in glob.glob(root + "/logs/*"):
        os.makedirs(WORK + "/logs", exist_ok=True); shutil.copy(d, WORK + "/logs/prev_" + os.path.basename(d))
print("train csv:", TRAIN_CSV, "| neighbours:", NN_PATH)
print("RESUMING from", RESUMED_FROM if RESUMED_FROM else "- (fresh run)")
'''),
        md("## 1. Build the training prompts (skipped when resuming: the previous session's prompts are reused byte for byte)"),
        code('''
from transformers import AutoTokenizer
import prompts as P
tok = AutoTokenizer.from_pretrained(MODEL)
rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
TR, DV = "%s/data/train_%s.jsonl" % (WORK, MODE), "%s/data/dev_%s.jsonl" % (WORK, MODE)
if not RESUMED_FROM:
    NN = json.load(op(NN_PATH))
    ctx = P.Context(rows)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"][:N_DEV]
    sub = train_rows if not N_TRAIN else random.Random(42).sample(train_rows, min(N_TRAIN, len(train_rows)))
    pool = [r["question_id"] for r in train_rows]
    missing = [r["question_id"] for r in sub if r["question_id"] not in NN["train"]]
    assert not missing, "%d training rows have no re-ranked neighbours" % len(missing)
    for path, items, split, noise in ((TR, sub, "train", NOISE), (DV, dev_rows, "dev", 0.0)):
        r = random.Random(1)
        with open(path, "w", encoding="utf-8") as f:
            for row in items:
                q = row["question_id"]
                ex = P.make_example(ctx, MODE, q, row["question"], NN[split][q], answer=row["final_answer"], noise=noise,
                                    rng=r, pool_qids=pool, tok=tok)
                f.write(json.dumps(ex, ensure_ascii=False) + "\\n")
ex = [json.loads(l) for l in open(TR, encoding="utf-8")]
lens = [len(tok(e["system"] + e["user"], add_special_tokens=False)["input_ids"]) for e in ex[:500]]
print("training examples: %d | dev: %d | prompt tokens (first 500): mean %.0f max %d" % (
    len(ex), sum(1 for _ in open(DV, encoding="utf-8")), sum(lens) / len(lens), max(lens)))
print("\\n--- example prompt ---\\n%s\\n--- target starts: %s" % (ex[0]["user"][:1500], ex[0]["answer"][:200].replace("\\n", " ")))
'''),
        md("## 2. Pilot on the 8 LONGEST examples: memory at MAX_LEN and speed, two GPUs (falls back to one if DDP fails)"),
        code('''
if not EXTRA:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)

def tl(e):
    return len(tok(e["system"] + e["user"] + e["answer"], add_special_tokens=False)["input_ids"])
longest = sorted(ex, key=tl, reverse=True)[:8]
with open(WORK + "/data/pilot.jsonl", "w", encoding="utf-8") as f:
    for e in longest:
        f.write(json.dumps(e, ensure_ascii=False) + "\\n")
print("pilot examples: %d-%d tokens (cut to MAX_LEN=%d)" % (tl(longest[-1]), tl(longest[0]), MAX_LEN))

ENV = dict(os.environ, TOKENIZERS_PARALLELISM="false", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", USE_LIBUV="0")

def launch(nproc, args, log=None):
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc)] if nproc > 1 else [sys.executable]
    env = dict(ENV, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(max(1, nproc)))) if not EXTRA else ENV
    if log is None:
        return subprocess.run(base + ["train_qlora.py"] + args + EXTRA, env=env, capture_output=True, text=True)
    return subprocess.Popen(base + ["train_qlora.py"] + args + EXTRA, env=env, stdout=log, stderr=subprocess.STDOUT)

def targs(train, out, accum, extra=()):
    return ["--train", train, "--dev", DV, "--out", out, "--model", MODEL, "--max_len", str(MAX_LEN), "--grad_accum", str(accum),
            "--epochs", str(EPOCHS), "--train_hours", str(TRAIN_HOURS), "--save_steps", str(SAVE_STEPS), "--eval_steps", "100"] + list(extra)

NPROC = max(1, N_GPU)
for attempt in ([NPROC, 1] if NPROC > 1 else [1]):
    shutil.rmtree(WORK + "/pilot", ignore_errors=True)
    r = launch(attempt, targs(WORK + "/data/pilot.jsonl", WORK + "/pilot", 2, ["--max_steps", "2", "--save_steps", "1000", "--eval_steps", "1000"]))
    if r.returncode == 0:
        NPROC = attempt
        break
    print("pilot with %d process(es) FAILED:\\n" % attempt, r.stdout[-2500:], r.stderr[-2500:])
else:
    raise AssertionError("pilot failed on 1 GPU too - read the error above (out of memory: lower MAX_LEN to 2560)")
s = json.load(open(WORK + "/pilot/train_summary.json"))
GRAD_ACCUM = EFFECTIVE_BATCH // NPROC
per_ex = s["sec_per_step"] / s["examples_per_step"] * NPROC        # seconds per example on one GPU (longest examples)
done = 0
if RESUMED_FROM:
    done = int(RESUMED_FROM.split("-")[1]) * EFFECTIVE_BATCH
proj = per_ex * (len(ex) * EPOCHS - done) / NPROC / 3600
print("pilot OK with %d GPU process(es) | %.1f s per (longest) example per GPU | effective batch %d = %d x %d accumulation x %d GPU" % (
    NPROC, per_ex, EFFECTIVE_BATCH, 1, GRAD_ACCUM, NPROC))
print("upper-bound projection for the remaining %d examples: %.1f h (real examples are shorter; TRAIN_HOURS=%.1f)" % (
    len(ex) * EPOCHS - done, proj, TRAIN_HOURS))
if torch.cuda.is_available():
    print("peak GPU memory in the pilot is in its log; the run will use MAX_LEN=%d" % MAX_LEN)
shutil.rmtree(WORK + "/pilot", ignore_errors=True)
'''),
        md("## 3. Train (checkpoint every %d steps; resumes automatically)" % 25),
        code('''
os.makedirs(WORK + "/logs", exist_ok=True)
log = open("%s/logs/train_%s.log" % (WORK, MODE), "a")
p = launch(NPROC, targs(TR, OUT, GRAD_ACCUM), log=log)
t0 = time.time()
while p.poll() is None:
    time.sleep(180 if not EXTRA else 5)
    lines = [l for l in open("%s/logs/train_%s.log" % (WORK, MODE), errors="ignore").read().splitlines()
             if any(k in l for k in ("'loss'", "eval_loss", "DONE", "Error", "Traceback", "examples", "trainable"))]
    print("[%5.0f min] %s" % ((time.time() - t0) / 60, lines[-1][:200] if lines else "(starting)"), flush=True)
print("training process exit code", p.returncode)
if p.returncode:
    print(open("%s/logs/train_%s.log" % (WORK, MODE), errors="ignore").read()[-4000:])
'''),
        md("## 4. Result: finished or needs another session?"),
        code('''
s = json.load(open(OUT + "/train_summary.json"))
print({k: v for k, v in s.items() if k not in ("loss_log", "eval_log")})
print("train loss (every ~50 steps):", s["loss_log"][::10])
print("dev loss:", s["eval_log"])
total = s["steps_per_epoch"] * EPOCHS
print("\\nSTATUS: %s | step %d of ~%d" % (s["status"], s["steps"], total))
if s["status"] == "time_limit":
    print(">>> NOT FINISHED. Run this notebook again with THIS notebook's output attached (Add Input -> Your Work); it resumes from the last checkpoint.")
elif s["status"] == "completed":
    print(">>> FINISHED. The adapter is in adapters/%s/final. Attach this notebook's output to the inference notebook." % MODE)
'''),
        md("## 5. Sanity check: 3 dev answers with the fixed decoding (answer-only penalty 1.1 + no repeated 12-token n-gram)"),
        code('''
if EXTRA or s["status"] != "completed":
    print("skipped (smoke test or unfinished run)")
else:
    import infer
    from types import SimpleNamespace
    ga = SimpleNamespace(no_quant=False, model=MODEL, adapter_dir=WORK + "/adapters", max_new=1280, rep_penalty=1.0, gen_rep_penalty=1.1,
                         gen_no_repeat=12, do_sample=False, temperature=0.7, top_p=0.9, seed=0)
    model, _ = infer.load_model(SimpleNamespace(no_quant=False, model=MODEL, adapter_dir=OUT), ["final"])
    tok.padding_side = "left"
    dev = [json.loads(l) for l in open(DV, encoding="utf-8")][:3]
    texts = [tok.apply_chat_template([{"role": "system", "content": e["system"]}, {"role": "user", "content": e["user"]}],
                                     tokenize=False, add_generation_prompt=True) for e in dev]
    for e, (t, n) in zip(dev, infer.generate(model, tok, texts, ga, "cuda:0")):
        print("=" * 90, "\\nQ:", e["user"].split("Question: ")[-1][:160], "\\n", t[:900], "\\n... (%d tokens, %d words)" % (n, len(t.split())))
'''),
        md("""
## After it finishes
* **Section 4 says NOT FINISHED:** open this notebook -> **Add Input -> Your Work -> this notebook** (its latest version) -> **Save & Run All** again.
  It copies the last checkpoint and the exact training prompts from that output and continues.
* **Section 4 says FINISHED:** send me the notebook link. The adapter is `adapters/nn/final`; the evaluation/inference notebook attaches this output directly.
"""),
    ]
    return notebook(cells)


def build_final_eval_nb():
    def src(name):
        return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    cells = [
        md("""
# 09 - Final inference and evaluation: every system, same questions, same decoding

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
> This is the **inference notebook** of the submission: it loads the trained LoRA checkpoints, answers the hold-out test questions and the
> 1,000 unseen questions (Xt), and scores every system with the shared harness. No training.

**Inputs (Add Input):**
1. `train_processed.csv` (dataset `train-processed1`) and `test_processed.csv` (dataset `train-processed`: the unseen 1,000 questions)
2. `question_neighbours.json` (dataset `nlp-qna-retrieval`: original retrieval order)
3. `question_neighbours_reranked.json.gz` (Your Work -> re-ranker notebook 07 output)
4. the Phase 4 adapters (dataset `nlp-qna-adapters`: `nn/`, `none/`)
5. the retrained adapter (Your Work -> retrain notebook 08 output: `adapters/nn/final`)

**Systems on the hold-out test questions** (the same seed-42 sample of 300 as Phase 5; questions never seen in training):

| system | model | reference answers in the prompt |
|---|---|---|
| B0a (original) | none - copy the nearest training answer | - |
| B0a (re-ranked) | none - copy the top re-ranked training answer | - |
| B1 | Qwen2.5-3B-Instruct zero-shot | none |
| B2 | Phase 4 adapter `none` | none |
| P1 | Phase 4 adapter `nn` (4,000 rows) | top-2, original order |
| **P2 (final)** | retrained adapter (all 12,825 rows) | **top-2, re-ranked** |
| P2 ablation | retrained adapter | top-2, original order |

All generated systems use the **same decoding**, chosen in the decoding sweep (notebook 06): greedy, repetition penalty 1.1 on the answer's own tokens,
no repeated 12-token n-gram inside the answer, at most 1,280 new tokens.

**Unseen set (Xt):** exact repeats of a training question -> stored medoid answer (router); every other question -> P2.
Writes `xt_predictions.csv`, `xt_predictions_detailed.csv`, `xt_predictions_with_baseline.csv` and `metrics.json`.
"""),
        code("!pip install -q -U peft bitsandbytes accelerate huggingface_hub rouge-score sacrebleu"),
        code("%%writefile prompts.py\n" + src("prompts.py")),
        code("%%writefile eval_harness.py\n" + src("eval_harness.py")),
        code("%%writefile infer.py\n" + src("infer.py")),
        md("## 0. Configuration and inputs"),
        code('''
import glob, os, sys, json, gzip, csv, random, time, shutil, subprocess, collections as C
csv.field_size_limit(10 ** 9)
_missing = [f for f in ("prompts.py", "eval_harness.py", "infer.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - use Run All from the top"
sys.path.insert(0, os.getcwd())
import torch
CPU = bool(os.environ.get("P9_CPU"))      # CPU smoke-test hook, never set on Kaggle
if not CPU:
    assert torch.cuda.is_available(), "No GPU. Session options -> Accelerator -> GPU T4 x2, then Run All."
N_GPU = torch.cuda.device_count()
print("GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])

MODEL = os.environ.get("P9_MODEL", "Qwen/Qwen2.5-3B-Instruct")
N_HOLDOUT = int(os.environ.get("P9_NHOLD", "300"))   # same seed-42 sample as Phase 5; 0 = all 1,500 (about 5x slower)
ABLATION_ORIG_ORDER = True     # also run the final model with the ORIGINAL retrieval order (isolates the re-ranker at inference)
DECODE = {"max_new": int(os.environ.get("P9_MAXNEW", "1280")), "rep_penalty": 1.0, "gen_rep_penalty": 1.1, "gen_no_repeat": 12}
XT_LIMIT = int(os.environ.get("P9_XT_LIMIT", "0"))  # smoke tests only (0 = all)
BATCH = 10
GEN_HOURS = 10.5               # each worker stops cleanly after this; re-run to resume (finished answers are kept)
BERTSCORE = False
WORK = os.environ.get("P9_WORK", "/kaggle/working")
INPUT = os.environ.get("P9_INPUT", "/kaggle/input")

def find(pattern, required=True):
    hit = sorted(p.replace("\\\\", "/") for p in glob.glob(INPUT + "/**/" + pattern, recursive=True))
    if not hit and required:
        raise AssertionError("attach a dataset / notebook output containing " + pattern)
    return hit[0] if hit else None

def op(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")

TRAIN_CSV, UNSEEN_CSV = find("train_processed*.csv"), find("test_processed*.csv")
NN_ORIG, NN_RR = find("question_neighbours.json*"), find("question_neighbours_reranked.json*")
V1_NN = os.path.dirname(find("nn/adapter_config.json"))
V1_NONE = os.path.dirname(find("none/adapter_config.json"))
V2_NN = os.path.dirname(find("adapters/nn/final/adapter_config.json"))
V2_SUMMARY = os.path.join(os.path.dirname(V2_NN), "train_summary.json")
V2_STATUS = json.load(open(V2_SUMMARY)) if os.path.exists(V2_SUMMARY) else {}
print("retrained adapter:", V2_NN, "| training status:", V2_STATUS.get("status"), "| steps:", V2_STATUS.get("steps"),
      "of", V2_STATUS.get("steps_per_epoch"))
if V2_STATUS.get("status") != "completed":
    print("WARNING: the retrained adapter did not finish its epoch - results will describe a PARTIALLY trained model.")

AD = WORK + "/ad"                                  # one folder with all three adapters under distinct names
for name, srcdir in (("nn_v1", V1_NN), ("none_v1", V1_NONE), ("nn_v2", V2_NN)):
    if not os.path.exists(AD + "/" + name):
        shutil.copytree(srcdir, AD + "/" + name)
print("adapters:", {n: len(os.listdir(AD + "/" + n)) for n in os.listdir(AD)}, "| decoding:", DECODE, "| hold-out N:", N_HOLDOUT)
'''),
        md("## 1. Router and prompts (CPU)"),
        code('''
import prompts as P
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
unseen = list(csv.DictReader(open(UNSEEN_CSV, encoding="utf-8")))
NNO, NNR = json.load(op(NN_ORIG)), json.load(op(NN_RR))
ctx = P.Context(rows)
qid2row = ctx.qid2row

# exact-match router: unseen question text == a training question -> that group's medoid answer
medoid = {r["group_id"]: r for r in rows if r["is_medoid"] == "1"}
routed = {u["question_id"]: medoid[u["matched_group_id"]]["final_answer"]
          for u in unseen if u["in_train"] == "1" and u["matched_group_id"] in medoid}
gen_unseen = [u for u in unseen if u["question_id"] not in routed]
if XT_LIMIT:
    gen_unseen = gen_unseen[:XT_LIMIT]
print("unseen: %d | routed by exact match: %d (%.1f%%) | to generate: %d" % (len(unseen), len(routed), 100 * len(routed) / len(unseen), len(gen_unseen)))

test_rows = [r for r in rows if r["split"] == "test"]
hold = random.Random(42).sample(test_rows, min(N_HOLDOUT, len(test_rows))) if N_HOLDOUT else test_rows
print("hold-out rows (never seen in training):", len(hold), "of", len(test_rows))

# (name, adapter, uid tag, prompt mode, neighbour lists)
SYSTEMS = [("B1 zero-shot base model", "base", "hold", "none", None),
           ("B2 fine-tuned v1, no retrieval", "none_v1", "hold", "none", None),
           ("P1 fine-tuned v1 + retrieval (Phase 4)", "nn_v1", "hold", "nn", NNO),
           ("P2 FINAL fine-tuned v2 + re-ranked retrieval", "nn_v2", "hold", "nn", NNR)]
if ABLATION_ORIG_ORDER:
    SYSTEMS.append(("P2 ablation: original retrieval order", "nn_v2", "hold_orig", "nn", NNO))
FINAL = "P2 FINAL fine-tuned v2 + re-ranked retrieval"

def mk(adapter, prio, tag, qid, question, mode, NNs, split):
    ex = P.make_example(ctx, mode, qid, question, NNs[split].get(qid, []) if mode != "none" else [], tok=tok)
    return {"uid": "%s|%s|%s" % (adapter, tag, qid), "adapter": adapter, "prio": prio, "system": ex["system"], "user": ex["user"]}

jobs = []
for r in hold:
    for name, ad, tag, mode, NNs in SYSTEMS:
        jobs.append(mk(ad, 0, tag, r["question_id"], r["question"], mode, NNs, "test"))
for u in gen_unseen:
    jobs.append(mk("nn_v2", 1, "xt", u["question_id"], u["question"], "nn", NNR, "xt"))
os.makedirs(WORK + "/gen", exist_ok=True)
with open(WORK + "/jobs.jsonl", "w", encoding="utf-8") as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + "\\n")
print("jobs:", len(jobs), dict(C.Counter(j["adapter"] for j in jobs)))
'''),
        md("## 2. Generate - one worker per GPU (resumable)"),
        code('''
if not CPU:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
workers = [["nn_v2"], ["base", "none_v1", "nn_v1"]] if N_GPU >= 2 else [["base", "none_v1", "nn_v1", "nn_v2"]]
procs = []
for gpu, names in enumerate(workers):
    log = open("%s/gen/worker%d.log" % (WORK, gpu), "a")
    cmd = [sys.executable, "infer.py", "--jobs", WORK + "/jobs.jsonl", "--adapter_dir", AD, "--adapters", ",".join(names),
           "--out", "%s/gen/gen_gpu%d.jsonl" % (WORK, gpu), "--model", MODEL, "--batch", str(BATCH), "--hours", str(GEN_HOURS),
           "--max_new", str(DECODE["max_new"]), "--rep_penalty", str(DECODE["rep_penalty"]),
           "--gen_rep_penalty", str(DECODE["gen_rep_penalty"]), "--gen_no_repeat", str(DECODE["gen_no_repeat"])]
    if CPU:
        cmd.append("--no_quant")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
    if not CPU:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    print("started worker", gpu, names)
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(120 if not CPU else 3)
    for gpu, p in procs:
        lines = open("%s/gen/worker%d.log" % (WORK, gpu), errors="ignore").read().splitlines()
        print("[%4.0f min] worker %d: %s" % ((time.time() - t0) / 60, gpu, lines[-1][:150] if lines else "(starting)"), flush=True)
for gpu, p in procs:
    print("worker", gpu, "exit code", p.returncode)
    if p.returncode:
        print(open("%s/gen/worker%d.log" % (WORK, gpu), errors="ignore").read()[-3000:])
'''),
        md("## 3. Collect the answers"),
        code('''
import re
gen = {}
for f in glob.glob(WORK + "/gen/gen_gpu*.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); gen[d["uid"]] = d
missing = [j["uid"] for j in jobs if j["uid"] not in gen]
print("generated: %d of %d jobs | missing: %d" % (len(gen), len(jobs), len(missing)))
if missing:
    print("WARNING: jobs missing (a worker stopped early) - run the notebook again; it resumes. First:", missing[:3])
WORD = re.compile(r"\\S+")
def is_loop(text, n=8, min_rep=3):
    w = [x.lower().strip(".,;:*#-()") for x in WORD.findall(text)]
    c = C.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return bool(c) and max(c.values()) >= min_rep
'''),
        md("## 4. Hold-out evaluation - the main results table"),
        code('''
import numpy as np, pandas as pd
import eval_harness as H
def nearest(NNs, split, q):
    nb = NNs[split].get(q)
    return qid2row[nb[0][0]]["final_answer"] if nb else ""

ids = [r["question_id"] for r in hold if all(("%s|%s|%s" % (ad, tag, r["question_id"])) in gen for _, ad, tag, _, _ in SYSTEMS)]
refs = {r["question_id"]: r["final_answer"] for r in hold if r["question_id"] in set(ids)}
systems = {"B0a copy nearest answer (original order)": {q: nearest(NNO, "test", q) for q in ids},
           "B0a copy nearest answer (re-ranked)": {q: nearest(NNR, "test", q) for q in ids}}
extra = {n: {} for n in systems}
for name, ad, tag, mode, NNs in SYSTEMS:
    p, e, cap = {}, 0, 0
    for q in ids:
        d = gen["%s|%s|%s" % (ad, tag, q)]
        cap += d["hit_limit"]
        if d["text"].strip():
            p[q] = d["text"]
        else:
            p[q], e = nearest(NNR, "test", q), e + 1     # empty generation -> nearest answer (counted)
    systems[name] = p
    extra[name] = {"empty_replaced": e, "hit_cap": cap}
res = {}
for n, p in systems.items():
    res[n] = H.evaluate(p, refs, bertscore=BERTSCORE)
    res[n].update(extra[n])
    res[n]["loops_8gram_x3"] = sum(is_loop(p[q]) for q in ids)
    res[n]["median_words"] = float(np.median([len(p[q].split()) for q in ids]))
cols = ["rougeL", "rouge1", "rouge2", "bleu", "fact_f1", "len_ratio_mean", "median_words", "hit_cap", "loops_8gram_x3", "empty_replaced"] + (["bertscore_f1"] if BERTSCORE else [])
table = pd.DataFrame({n: {c: r.get(c, np.nan) for c in cols} for n, r in res.items()}).T
pd.set_option("display.width", 250)
print("hold-out questions scored: %d (reference median %d words)" % (len(ids), np.median([len(refs[q].split()) for q in ids])))
print(table.round(4).to_string())

from rouge_score import rouge_scorer
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
per = {n: np.array([sc.score(refs[q], p[q])["rougeL"].fmeasure for q in ids]) for n, p in systems.items()}
idx = np.random.default_rng(0).integers(0, len(ids), (2000, len(ids)))
def diff_ci(a, b):
    d = (per[a] - per[b])[idx].mean(axis=1)
    return "%+.4f  [95%% CI %+.4f, %+.4f]" % (per[a].mean() - per[b].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
B0O, B0R = "B0a copy nearest answer (original order)", "B0a copy nearest answer (re-ranked)"
pairs = [(FINAL, B0R, "final - copy re-ranked nearest answer (value of generating)"),
         (FINAL, B0O, "final - copy nearest answer, original order"),
         (FINAL, "P1 fine-tuned v1 + retrieval (Phase 4)", "final - Phase 4 model (value of retraining + re-ranking)"),
         (FINAL, "B2 fine-tuned v1, no retrieval", "final - no retrieval"),
         (FINAL, "B1 zero-shot base model", "final - zero-shot"),
         ("P1 fine-tuned v1 + retrieval (Phase 4)", B0O, "Phase 4 model - copy nearest (original order)"),
         (B0R, B0O, "re-ranked copy - original copy (value of the re-ranker alone)")]
if ABLATION_ORIG_ORDER:
    pairs.append((FINAL, "P2 ablation: original retrieval order", "final - same model with original order (re-ranker at inference)"))
boot = {}
print("\\nROUGE-L paired bootstrap (n=%d, 2000 resamples):" % len(ids))
for a, b, label in pairs:
    boot[label] = diff_ci(a, b)
    print("  %-70s %s" % (label, boot[label]))
'''),
        md("## 5. The router, scored separately\nOnly repeated questions can be routed, so this is measured on training questions that appear more than once: the stored answer is compared with a *different* human answer to the same question."),
        code('''
groups = C.defaultdict(list)
for r in rows:
    groups[r["group_id"]].append(r)
rep = [g for g in groups.values() if len(g) >= 2]
rr = random.Random(7)
held, p_med, p_rand = {}, {}, {}
for g in rr.sample(rep, min(1500, len(rep))):
    med = next((x for x in g if x["is_medoid"] == "1"), g[0])
    h = rr.choice([x for x in g if x is not med])
    held[h["question_id"]] = h["final_answer"]
    p_med[h["question_id"]] = med["final_answer"]
    p_rand[h["question_id"]] = rr.choice([x for x in g if x is not h])["final_answer"]
router_res, rand_res = H.evaluate(p_med, held), H.evaluate(p_rand, held)
print("repeated-question groups: %d | scored: %d" % (len(rep), len(held)))
print("router (stored medoid answer): rougeL %.4f | random other human answer: rougeL %.4f (conservative)" % (router_res["rougeL"], rand_res["rougeL"]))
share = len(routed) / len(unseen)
print("\\nESTIMATE for the unseen set (NOT a measurement: mixes a training-set router check with the hold-out score):")
for k in ("rougeL", "rouge1", "fact_f1"):
    print("   %-8s %.4f  (%.1f%% routed x router + %.1f%% generated x final model)" % (k, share * rand_res[k] + (1 - share) * res[FINAL][k], 100 * share, 100 * (1 - share)))
'''),
        md("## 6. Write the predictions for the unseen set"),
        code('''
final, source, baseline, fallback = {}, {}, {}, 0
for u in unseen:
    q = u["question_id"]
    if q in routed:
        final[q] = baseline[q] = routed[q]; source[q] = "router"
        continue
    baseline[q] = nearest(NNR, "xt", q)       # baseline system: same router, then copy the re-ranked nearest answer
    d = gen.get("nn_v2|xt|%s" % q)
    if d and d["text"].strip():
        final[q], source[q] = d["text"], "generated"
    else:
        final[q], source[q] = baseline[q], "fallback_nearest"
        fallback += 1
assert len(final) == len(unseen) == 1000 and all(v.strip() for v in final.values()), "predictions incomplete"
def write(path, header, rowf):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for u in unseen:
            w.writerow(rowf(u))
write(WORK + "/xt_predictions.csv", ["question_id", "question", "final_answer"], lambda u: [u["question_id"], u["question"], final[u["question_id"]]])
write(WORK + "/xt_predictions_detailed.csv", ["question_id", "source", "final_answer"], lambda u: [u["question_id"], source[u["question_id"]], final[u["question_id"]]])
write(WORK + "/xt_predictions_with_baseline.csv", ["question_id", "question", "final_answer", "baseline_answer"],
      lambda u: [u["question_id"], u["question"], final[u["question_id"]], baseline[u["question_id"]]])
print("wrote xt_predictions*.csv:", len(final), "rows |", dict(C.Counter(source.values())))
if fallback:
    print("WARNING: %d unseen questions used the nearest-answer fallback (missing generations) - re-run to resume." % fallback)
xt_gen = [gen[u] for u in gen if u.startswith("nn_v2|xt|")]
json.dump({"holdout_n": len(ids), "decoding": DECODE, "systems": res, "bootstrap_rougeL": boot,
           "router": router_res, "router_random_other": rand_res, "routed_share": share,
           "unseen": {"sources": dict(C.Counter(source.values())), "generated_hit_cap": sum(d["hit_limit"] for d in xt_gen),
                      "generated_loops_8gram_x3": sum(is_loop(d["text"]) for d in xt_gen)},
           "retrained_adapter_training": {k: v for k, v in V2_STATUS.items() if k not in ("loss_log", "eval_log")},
           "config": {"MODEL": MODEL, "N_HOLDOUT": N_HOLDOUT, "ABLATION_ORIG_ORDER": ABLATION_ORIG_ORDER}},
          open(WORK + "/metrics.json", "w"), indent=1, default=float)
print("wrote metrics.json")
'''),
        md("## 7. A few answers side by side (for the report's qualitative section)"),
        code('''
for q in ids[:3]:
    print("=" * 100, "\\nQ:", qid2row[q]["question"][:200])
    print("--- reference :", refs[q][:400].replace("\\n", " "))
    print("--- final (P2):", systems[FINAL][q][:400].replace("\\n", " "))
    print("--- B0a re-ranked copy:", systems[B0R][q][:250].replace("\\n", " "))
'''),
        md("""
## After it finishes
Send me the notebook link (I download the output with the Kaggle CLI). Output tab: `xt_predictions.csv` (the deliverable),
`xt_predictions_with_baseline.csv`, `xt_predictions_detailed.csv`, `metrics.json`, `gen/` (every raw answer), `jobs.jsonl` (every prompt).
"""),
    ]
    return notebook(cells)


def main():
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    p = os.path.join(ROOT, "notebooks", "09_final_inference_and_evaluation.ipynb")
    json.dump(build_final_eval_nb(), open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p)
    p = os.path.join(ROOT, "notebooks", "08_retrain_full.ipynb")
    json.dump(build_retrain_nb(), open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p)
    p = os.path.join(ROOT, "notebooks", "07_rerank_neighbours.ipynb")
    json.dump(build_rerank_nb(), open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p)
    if os.path.exists(SWEEP_PAYLOAD):
        p = os.path.join(ROOT, "notebooks", "06_decoding_sweep.ipynb")
        json.dump(build_decode_sweep_nb(), open(p, "w", encoding="utf-8"), indent=1)
        print("wrote", p)
    for name, nb in (("01_download_extract_pdfs.ipynb", build_download_nb()),
                     ("00_phase0_data_foundation.ipynb", build_phase0_nb()),
                     ("02_quality_and_chunking.ipynb", build_phase1_nb()),
                     ("03_retrieval_and_b0.ipynb", build_phase2_nb()),
                     ("04_train_qlora.ipynb", build_phase4_nb()),
                     ("05_inference_and_evaluation.ipynb", build_phase5_nb())):
        p = os.path.join(ROOT, "notebooks", name)
        json.dump(nb, open(p, "w", encoding="utf-8"), indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()
