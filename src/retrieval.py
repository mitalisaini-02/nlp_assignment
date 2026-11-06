"""Phase 2 retrieval helpers: dense search, reciprocal-rank fusion, same-group exclusion, ROUGE-L pair scoring,
chunk-recall metrics. Pure ASCII source on purpose."""
import json
import numpy as np
from multiprocessing import Pool


# ----------------------------------------------------------------------------- dense search
def dense_topk(Q, D, k, batch=512):
    """Q (nq, d) and D (nd, d) are L2-normalised. Returns (idx, score) of the top-k rows of D for every query."""
    k = min(k, D.shape[0])
    nq = Q.shape[0]
    idx = np.zeros((nq, k), dtype=np.int32)
    sc = np.zeros((nq, k), dtype=np.float32)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        Dt = torch.from_numpy(np.ascontiguousarray(D)).to(dev, dtype=dt)
        for i in range(0, nq, batch):
            q = torch.from_numpy(np.ascontiguousarray(Q[i:i + batch])).to(dev, dtype=dt)
            v, ix = (q @ Dt.T).float().topk(k, dim=1)
            idx[i:i + batch] = ix.cpu().numpy()
            sc[i:i + batch] = v.cpu().numpy()
        return idx, sc
    except ImportError:
        pass
    D32 = D.astype(np.float32)
    for i in range(0, nq, batch):
        S = Q[i:i + batch].astype(np.float32) @ D32.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-ps, axis=1)
        idx[i:i + batch] = np.take_along_axis(part, order, axis=1)
        sc[i:i + batch] = np.take_along_axis(ps, order, axis=1)
    return idx, sc


# ----------------------------------------------------------------------------- fusion
def rrf(rank_lists, weights, k=60, topn=50):
    """Weighted reciprocal-rank fusion. rank_lists: list of (nq, depth) int arrays of candidate ids (-1 = none).
    Returns (ids, scores) of shape (nq, topn); ids padded with -1."""
    nq = rank_lists[0].shape[0]
    ids = np.full((nq, topn), -1, dtype=np.int32)
    scs = np.zeros((nq, topn), dtype=np.float32)
    for q in range(nq):
        acc = {}
        for L, w in zip(rank_lists, weights):
            for r, c in enumerate(L[q]):
                if c >= 0:
                    acc[c] = acc.get(c, 0.0) + w / (k + r + 1)
        best = sorted(acc.items(), key=lambda kv: -kv[1])[:topn]
        for j, (c, s) in enumerate(best):
            ids[q, j], scs[q, j] = c, s
    return ids, scs


def drop_groups(ids, scores, pool_group, query_group, topn):
    """Remove candidates whose group equals the query's group (a question must not retrieve itself or its repeats)."""
    out_i = np.full((ids.shape[0], topn), -1, dtype=np.int32)
    out_s = np.zeros((ids.shape[0], topn), dtype=np.float32)
    for q in range(ids.shape[0]):
        j = 0
        for c, s in zip(ids[q], scores[q]):
            if c < 0 or pool_group[c] == query_group[q]:
                continue
            out_i[q, j], out_s[q, j] = c, s
            j += 1
            if j == topn:
                break
    return out_i, out_s


# ----------------------------------------------------------------------------- ROUGE-L on many pairs
_SC = None
CAP_WORDS = 700  # long answers are truncated for selection/tuning (LCS is quadratic); final metrics use full text


def _rl(pair):
    global _SC
    if _SC is None:
        from rouge_score import rouge_scorer
        _SC = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    a, b = pair
    return _SC.score(" ".join(a.split()[:CAP_WORDS]), " ".join(b.split()[:CAP_WORDS]))["rougeL"].fmeasure


def rougeL_pairs(pairs, procs=4):
    if not pairs:
        return []
    with Pool(procs) as p:
        return p.map(_rl, pairs, chunksize=16)


def mbr_pick(answer_lists, procs=4):
    """For each query pick the candidate answer with the highest mean ROUGE-L against the other candidates
    (the 'medoid'). answer_lists: list of lists of strings. Returns the chosen position per query."""
    pairs, owner = [], []
    for qi, al in enumerate(answer_lists):
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                pairs.append((al[i], al[j]))
                owner.append((qi, i, j))
    scores = rougeL_pairs(pairs, procs)
    tot = [[0.0] * len(al) for al in answer_lists]
    for (qi, i, j), s in zip(owner, scores):
        tot[qi][i] += s
        tot[qi][j] += s
    return [int(np.argmax(t)) if t else 0 for t in tot]


# ----------------------------------------------------------------------------- chunk retrieval metrics
def chunk_gold(citations_json, chunk_pages, chunk_docs):
    """Gold sets for one question: pages (doc, page) and docs cited by the training answer that exist as chunks."""
    pages, docs = set(), set()
    for c in json.loads(citations_json):
        if c.get("kind") != "zoho" or c["doc_id"] not in chunk_docs:
            continue
        docs.add(c["doc_id"])
        for p in c.get("pages", []):
            if (c["doc_id"], p) in chunk_pages:
                pages.add((c["doc_id"], p))
    return pages, docs


def chunk_recall(hits, gold_pages, gold_docs, chunk_doc, chunk_page, ms=(1, 5, 10, 20)):
    """hits: (nq, >=max(ms)) chunk ids. Returns {metric: value} over the questions that have gold of that kind."""
    out = {}
    for m in ms:
        pg_h = pg_n = dc_h = dc_n = 0
        for q in range(hits.shape[0]):
            got = [int(c) for c in hits[q, :m] if c >= 0]
            if gold_pages[q]:
                pg_n += 1
                pg_h += any((chunk_doc[c], chunk_page[c]) in gold_pages[q] for c in got)
            if gold_docs[q]:
                dc_n += 1
                dc_h += any(chunk_doc[c] in gold_docs[q] for c in got)
        out["page_hit@%d" % m] = pg_h / max(1, pg_n)
        out["doc_hit@%d" % m] = dc_h / max(1, dc_n)
        out["n_page_q"], out["n_doc_q"] = pg_n, dc_n
    return out
