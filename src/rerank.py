"""Neighbour re-ranker: re-orders the top-10 retrieved training questions so that the ones whose ANSWER best matches the
query's (unknown) answer come first. Pure ASCII source on purpose.

Label   : ROUGE-L F1 between a neighbour's answer and the query's true answer (exact, bit-parallel LCS on rouge-score tokens).
Features: retrieval rank/score, idf-weighted question overlap, answer lengths, answer consensus among the 10 neighbours,
          group size / medoid flag, optional cross-encoder scores (question-question, question-answer).
Models  : sklearn HistGradientBoosting (pointwise, label relative to the query mean) and, if installed, LightGBM LambdaRank.
          Chosen on dev; reported on the hold-out test split. Train-split queries are re-ranked by 5-fold cross-fitting
          (folds by question group), so their prompts are never chosen with a model that saw their own answer.

    python rerank.py --train_csv train_processed.csv --unseen_csv test_processed.csv --nn question_neighbours.json.gz --out OUTDIR
"""
import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

csv.field_size_limit(10 ** 9)
_SC = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
T0 = time.time()


def log(*a):
    print("[%6.1f min]" % ((time.time() - T0) / 60), *a, flush=True)


def op(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


# ---------------------------------------------------------------- exact ROUGE-L via bit-parallel LCS
class Tok:
    def __init__(self):
        self.vocab, self.cache = {}, {}

    def ids(self, key, text):
        r = self.cache.get(key)
        if r is None:
            r = [self.vocab.setdefault(t, len(self.vocab)) for t in _SC._tokenizer.tokenize(text)]
            self.cache[key] = r
        return r


def masks(a):
    M = {}
    for i, x in enumerate(a):
        M[x] = M.get(x, 0) | (1 << i)
    return M


def lcs_len(Ma, la, b):
    """Hyyro (2004) bit-parallel LCS length; Ma = masks(a), la = len(a)."""
    if not la or not b:
        return 0
    full = (1 << la) - 1
    V = full
    get = Ma.get
    for y in b:
        U = V & get(y, 0)
        if U:
            V = ((V + U) | (V - U)) & full
    return la - bin(V).count("1")


def rl_f(Ma, la, b):
    lb = len(b)
    return 0.0 if not la or not lb else 2.0 * lcs_len(Ma, la, b) / (la + lb)


# ---------------------------------------------------------------- lexical features on questions
WORD = re.compile(r"[a-z][a-z0-9\-]+")


def cwords(t):
    return {w for w in WORD.findall(t.lower()) if len(w) > 2 and w not in ENGLISH_STOP_WORDS}


FEATS = ["rank", "score", "score_rel_top", "score_gap_next", "jaccard", "idf_cov_query", "idf_cov_nb", "max_shared_idf",
         "missing_idf", "log_ans_words", "log_nbq_words", "log_q_words", "consensus", "consensus_rank", "group_size", "is_medoid",
         "ce_q", "ce_q_rank", "ce_a", "ce_a_rank",
         "d_score", "d_idf_cov_query", "d_consensus", "d_ce_q", "d_ce_a"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--unseen_csv", required=True)
    p.add_argument("--nn", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ce_model", default="BAAI/bge-reranker-base")
    p.add_argument("--ce_q", action="store_true", help="cross-encoder score question vs neighbour question")
    p.add_argument("--ce_a", action="store_true", help="cross-encoder score question vs neighbour answer (slow on CPU)")
    p.add_argument("--max_train_q", type=int, default=0, help="smoke tests: subsample train queries")
    p.add_argument("--max_eval_q", type=int, default=0, help="smoke tests: subsample dev/test/xt queries")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.train_csv, encoding="utf-8")))
    unseen = list(csv.DictReader(open(a.unseen_csv, encoding="utf-8")))
    qid2row = {r["question_id"]: r for r in rows}
    NN = json.load(op(a.nn))
    log("rows %d | unseen %d | neighbour lists %s" % (len(rows), len(unseen), {s: len(v) for s, v in NN.items()}))

    # queries per split: (qid, question, true answer or None, group)
    Q = {}
    for s in ("train", "dev", "test"):
        Q[s] = [(r["question_id"], r["question"], r["final_answer"], r["group_id"]) for r in rows if r["split"] == s and r["question_id"] in NN[s]]
    Q["xt"] = [(u["question_id"], u["question"], None, "xt" + u["question_id"]) for u in unseen if u["question_id"] in NN["xt"]]
    rng = random.Random(0)
    if a.max_train_q:
        Q["train"] = rng.sample(Q["train"], min(a.max_train_q, len(Q["train"])))
    if a.max_eval_q:
        for s in ("dev", "test", "xt"):
            Q[s] = rng.sample(Q[s], min(a.max_eval_q, len(Q[s])))
    log("queries:", {s: len(v) for s, v in Q.items()})

    # idf over all question texts
    df = Counter()
    allq = [r["question"] for r in rows] + [u["question"] for u in unseen]
    for t in allq:
        df.update(cwords(t))
    N = len(allq)
    idf = lambda w: math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0  # noqa: E731

    # ---- sanity: the fast ROUGE-L equals rouge-score
    tok = Tok()
    chk = random.Random(1).sample(rows, 200)
    diffs = []
    for r1, r2 in zip(chk[::2], chk[1::2]):
        x, y = tok.ids(r1["question_id"], r1["final_answer"]), tok.ids(r2["question_id"], r2["final_answer"])
        diffs.append(abs(rl_f(masks(x), len(x), y) - _SC.score(r1["final_answer"], r2["final_answer"])["rougeL"].fmeasure))
    assert max(diffs) < 1e-9, "fast ROUGE-L differs from rouge-score: %g" % max(diffs)
    log("fast ROUGE-L matches rouge-score on 100 pairs (max diff %.1e)" % max(diffs))

    # ---- per-split feature matrices
    K = 10
    data = {}
    for s, qs in Q.items():
        nq = len(qs)
        X = np.full((nq, K, len(FEATS)), np.nan, dtype=np.float32)
        Y = np.full((nq, K), np.nan, dtype=np.float32)
        C = [[None] * K for _ in range(nq)]
        for i, (qid, qtext, ans, _) in enumerate(qs):
            cand = NN[s][qid][:K]
            qw = cwords(qtext)
            qidf = sum(idf(w) for w in qw) or 1.0
            ids_ = []
            for j, (nid, sc) in enumerate(cand):
                nr = qid2row[nid]
                C[i][j] = nid
                nw = cwords(nr["question"])
                sh = qw & nw
                f = X[i, j]
                f[0], f[1] = j, sc
                f[2] = sc / max(1e-9, cand[0][1])
                f[3] = sc - (cand[j + 1][1] if j + 1 < len(cand) else 0.0)
                f[4] = len(sh) / max(1, len(qw | nw))
                f[5] = sum(idf(w) for w in sh) / qidf
                f[6] = sum(idf(w) for w in sh) / (sum(idf(w) for w in nw) or 1.0)
                f[7] = max([idf(w) for w in sh], default=0.0)
                f[8] = sum(idf(w) for w in qw - nw)
                f[9] = math.log1p(int(nr.get("n_words_answer") or len(nr["final_answer"].split())))
                f[10] = math.log1p(len(nr["question"].split()))
                f[11] = math.log1p(len(qtext.split()))
                f[14] = float(nr.get("group_size") or 1)
                f[15] = float(nr.get("is_medoid") == "1")
                ids_.append(tok.ids(nid, nr["final_answer"]))
            Ms = [masks(x) for x in ids_]
            m = len(ids_)
            S = np.zeros((m, m), dtype=np.float32)
            for u in range(m):
                for v in range(u + 1, m):
                    S[u, v] = S[v, u] = rl_f(Ms[u], len(ids_[u]), ids_[v])
            cons = S.sum(1) / max(1, m - 1)
            X[i, :m, 12] = cons
            X[i, :m, 13] = np.argsort(np.argsort(-cons))
            if ans is not None:
                ta = tok.ids("ANS" + qid, ans)
                Mt = masks(ta)
                for j in range(m):
                    Y[i, j] = rl_f(Mt, len(ta), ids_[j])
            if (i + 1) % 2000 == 0:
                log("  %s %d/%d" % (s, i + 1, nq))
        data[s] = {"X": X, "Y": Y, "C": C, "q": qs}
        log("features %s done (%d queries)" % (s, nq))

    # ---- optional cross-encoder features
    if a.ce_q or a.ce_a:
        import torch
        from sentence_transformers import CrossEncoder
        dev_ = "cuda" if torch.cuda.is_available() else "cpu"
        ce = CrossEncoder(a.ce_model, max_length=384 if a.ce_a else 96, device=dev_)
        if dev_ == "cuda":
            ce.model.half()
        log("cross-encoder", a.ce_model, "on", dev_)
        for col, on in ((16, a.ce_q), (18, a.ce_a)):
            if not on:
                continue
            for s, d in data.items():
                pairs, where = [], []
                for i, (qid, qtext, _, _) in enumerate(d["q"]):
                    for j, nid in enumerate(d["C"][i]):
                        if nid is None:
                            continue
                        nr = qid2row[nid]
                        other = nr["question"] if col == 16 else " ".join(nr["final_answer"].split()[:220])
                        pairs.append((qtext, other))
                        where.append((i, j))
                sc = ce.predict(pairs, batch_size=128 if dev_ == "cuda" else 32, show_progress_bar=False)
                for (i, j), v in zip(where, sc):
                    d["X"][i, j, col] = float(v)
                Xc = d["X"][:, :, col]
                d["X"][:, :, col + 1] = np.argsort(np.argsort(-np.nan_to_num(Xc, nan=-1e9), axis=1), axis=1)
                log("cross-encoder %s %s: %d pairs" % ("q-q" if col == 16 else "q-answer", s, len(pairs)))
    # within-query deltas (value minus the best in the list)
    import warnings
    for d in data.values():
        X = d["X"]
        for dst, src in ((20, 1), (21, 5), (22, 12), (23, 16), (24, 18)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN when a cross-encoder feature is switched off
                X[:, :, dst] = X[:, :, src] - np.nanmax(X[:, :, src], axis=1, keepdims=True)
    used = [k for k, f in enumerate(FEATS) if not np.isnan(data["train"]["X"][:, :, k]).all()]
    log("features used (%d): %s" % (len(used), [FEATS[k] for k in used]))

    def flat(d):
        X = d["X"][:, :, used].reshape(-1, len(used))
        Y = d["Y"].reshape(-1)
        valid = np.array([[nid is not None for nid in c] for c in d["C"]]).reshape(-1)
        return X, Y, valid

    # ---- models
    from sklearn.ensemble import HistGradientBoostingRegressor
    try:
        import lightgbm as lgb
    except Exception:
        lgb = None

    def fit(name, X, Y, nq):
        Yq = Y.reshape(nq, K)
        rel = (Yq - np.nanmean(Yq, axis=1, keepdims=True)).reshape(-1)
        ok = ~np.isnan(rel)
        if name == "hgb":
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=0)
            m.fit(X[ok], rel[ok])
            return m
        # LambdaRank: graded relevance from the within-query rank of the label (0..9, best = 9)
        lab = np.argsort(np.argsort(np.nan_to_num(Yq, nan=-1.0), axis=1), axis=1).reshape(-1)
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                           reg_lambda=1.0, label_gain=list(range(K)), lambdarank_truncation_level=3, random_state=0, verbose=-1)
        m.fit(X, lab, group=[K] * nq)
        return m

    def predict(m, X, nq):
        return m.predict(X).reshape(nq, K)

    names = ["hgb"] + (["lgb_rank"] if lgb is not None else [])
    Xtr, Ytr, _ = flat(data["train"])
    ntr = len(data["train"]["q"])
    full = {n: fit(n, Xtr, Ytr, ntr) for n in names}
    log("models fitted on all train queries:", names)

    def top_metrics(order, Y):
        nq = Y.shape[0]
        Yz = np.nan_to_num(Y, nan=0.0)
        t1 = Yz[np.arange(nq), order[:, 0]]
        t2 = (Yz[np.arange(nq), order[:, 0]] + Yz[np.arange(nq), order[:, 1]]) / 2
        return t1, t2

    def order_by(P):
        return np.argsort(-np.nan_to_num(P, nan=-1e9), axis=1, kind="stable")

    report = {"n_queries": {s: len(d["q"]) for s, d in data.items()}, "features": [FEATS[k] for k in used], "eval": {}}
    preds = {}
    for s in ("dev", "test"):
        d = data[s]
        nq = len(d["q"])
        X, Y, _ = flat(d)
        orders = {"retrieval order (current)": np.tile(np.arange(K), (nq, 1)),
                  "oracle (best of 10)": order_by(d["Y"]),
                  "consensus only": order_by(d["X"][:, :, 12]),
                  "idf question overlap only": order_by(d["X"][:, :, 5])}
        if a.ce_q:
            orders["cross-encoder q-q only"] = order_by(d["X"][:, :, 16])
        if a.ce_a:
            orders["cross-encoder q-answer only"] = order_by(d["X"][:, :, 18])
        for n in names:
            P = predict(full[n], X, nq)
            preds[(s, n)] = P
            orders["model " + n] = order_by(P)
        res = {}
        for on, o in orders.items():
            t1, t2 = top_metrics(o, d["Y"])
            res[on] = {"top1_rougeL": float(t1.mean()), "top2_mean_rougeL": float(t2.mean()), "_t1": t1}
        report["eval"][s] = res
        log("%s (n=%d)" % (s, nq))
        for on, r in res.items():
            print("     %-30s top-1 RL %.4f | mean of top-2 %.4f" % (on, r["top1_rougeL"], r["top2_mean_rougeL"]), flush=True)

    best = max(names, key=lambda n: report["eval"]["dev"]["model " + n]["top1_rougeL"])
    report["chosen_model_on_dev"] = best
    t_new = report["eval"]["test"]["model " + best]["_t1"]
    t_old = report["eval"]["test"]["retrieval order (current)"]["_t1"]
    idx = np.random.default_rng(0).integers(0, len(t_new), (2000, len(t_new)))
    dd = (t_new - t_old)[idx].mean(axis=1)
    report["test_top1_gain_vs_retrieval"] = "%+.4f  [95%% CI %+.4f, %+.4f]" % (
        (t_new - t_old).mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5))
    log("chosen on dev: %s | hold-out test top-1 ROUGE-L gain vs current retrieval order: %s (n=%d)" % (
        best, report["test_top1_gain_vs_retrieval"], len(t_new)))

    # ---- cross-fitted predictions for train queries
    groups = sorted({q[3] for q in data["train"]["q"]})
    random.Random(42).shuffle(groups)
    fold_of = {g: i % a.folds for i, g in enumerate(groups)}
    fq = np.array([fold_of[q[3]] for q in data["train"]["q"]])
    Ptr = np.zeros((ntr, K), dtype=np.float32)
    X3 = data["train"]["X"][:, :, used]
    Y3 = data["train"]["Y"]
    for f in range(a.folds):
        tr, te = fq != f, fq == f
        m = fit(best, X3[tr].reshape(-1, len(used)), Y3[tr].reshape(-1), int(tr.sum()))
        Ptr[te] = predict(m, X3[te].reshape(-1, len(used)), int(te.sum()))
        log("  cross-fit fold %d/%d" % (f + 1, a.folds))
    o = order_by(Ptr)
    t1, _ = top_metrics(o, Y3)
    t1o, _ = top_metrics(np.tile(np.arange(K), (ntr, 1)), Y3)
    report["train_crossfit_top1_rougeL"] = {"retrieval": float(t1o.mean()), "reranked": float(t1.mean())}
    log("train (cross-fitted) top-1 RL: retrieval %.4f -> reranked %.4f" % (t1o.mean(), t1.mean()))

    # ---- write re-ranked neighbour lists (same format as question_neighbours.json)
    out = {}
    for s, d in data.items():
        nq = len(d["q"])
        if s == "train":
            P = Ptr
        elif (s, best) in preds:
            P = preds[(s, best)]
        else:
            X, _, _ = flat(d)
            P = predict(full[best], X, nq)
        O = order_by(P)
        out[s] = {d["q"][i][0]: [[d["C"][i][j], round(float(P[i, j]), 5)] for j in O[i] if d["C"][i][j] is not None] for i in range(nq)}
    with gzip.open(os.path.join(a.out, "question_neighbours_reranked.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(out, f)
    for s in report["eval"]:
        for r in report["eval"][s].values():
            r.pop("_t1", None)
    json.dump(report, open(os.path.join(a.out, "rerank_report.json"), "w"), indent=1)
    log("wrote question_neighbours_reranked.json.gz and rerank_report.json to", a.out)
    return report


if __name__ == "__main__":
    main()
