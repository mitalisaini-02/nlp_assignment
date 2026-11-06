"""Diagnostic: when does the generated answer beat copying the top re-ranked training answer, and can a gate that
only uses inference-time features (never the reference) pick the better one per question?

Reads the notebook 09 outputs (gen/*.jsonl, jobs.jsonl) and the neighbour files; writes diag_gate.json. CPU only.
    python -I src/diag_gate.py "inference results second run" C:/nlp_data/diag/diag_gate.json
"""
import csv, gzip, json, os, random, re, sys
import numpy as np
from rouge_score import rouge_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "inference results second run")
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/nlp_data/diag/diag_gate.json"
NN_ORIG = "C:/Users/ARTH/Downloads/question_neighbours.json.gz"
NN_RR = "C:/nlp_data/kaggle_out/p7/question_neighbours_reranked.json.gz"

rows = list(csv.DictReader(open(os.path.join(PROJ, "data/processed/train_processed.csv"), encoding="utf-8")))
qid2row = {r["question_id"]: r for r in rows}
test_rows = [r for r in rows if r["split"] == "test"]
hold = random.Random(42).sample(test_rows, 300)                     # identical to notebook 09
NNO = json.load(gzip.open(NN_ORIG, "rt", encoding="utf-8"))
NNR = json.load(gzip.open(NN_RR, "rt", encoding="utf-8"))

gen = {}
for f in ("gen_gpu0.jsonl", "gen_gpu1.jsonl"):
    for l in open(os.path.join(RUN, "gen", f), encoding="utf-8"):
        d = json.loads(l)
        gen[d["uid"]] = d
job_hold = {j.split("|")[2] for j in (json.loads(l)["uid"] for l in open(os.path.join(RUN, "jobs.jsonl"), encoding="utf-8")) if "|hold|" in j}
ids = [r["question_id"] for r in hold]
assert set(ids) == job_hold, "hold-out sample differs from notebook 09"

sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rl = lambda ref, hyp: sc.score(ref, hyp)["rougeL"].fmeasure
def is_loop(t):                                                      # an 8-gram that occurs 3 times
    w = re.findall(r"\w+", t.lower())
    c = {}
    for i in range(len(w) - 7):
        g = tuple(w[i:i + 8]); c[g] = c.get(g, 0) + 1
        if c[g] >= 3:
            return True
    return False
tokset = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
jac = lambda a, b: len(a & b) / max(1, len(a | b))

R = []
for r in hold:
    q = r["question_id"]
    nr, no = NNR["test"][q], NNO["test"][q]
    top, top2 = qid2row[nr[0][0]], qid2row[nr[1][0]]
    copy, g = top["final_answer"], gen["nn_v2|hold|%s" % q]
    ref = r["final_answer"]
    orig_score = dict((a, s) for a, s in no)
    R.append(dict(
        q=q, rl_gen=rl(ref, g["text"]), rl_copy=rl(ref, copy), rl_copy_orig=rl(ref, qid2row[no[0][0]]["final_answer"]),
        rl_p1=rl(ref, gen["nn_v1|hold|%s" % q]["text"]),
        # ---- inference-time features only ----
        rr_top=nr[0][1], rr_margin=nr[0][1] - nr[1][1],
        ret_score_top=orig_score.get(nr[0][0], 0.0), ret_top1_same=float(nr[0][0] == no[0][0]),
        q_jac=jac(tokset(r["question"]), tokset(top["question"])),
        gen_vs_copy=rl(copy, g["text"]), copy_vs_top2=rl(top2["final_answer"], copy),
        gen_words=len(g["text"].split()), copy_words=len(copy.split()),
        hit_cap=float(g["hit_limit"]), loop=float(is_loop(g["text"]))))
F = ["rr_top", "rr_margin", "ret_score_top", "ret_top1_same", "q_jac", "gen_vs_copy", "copy_vs_top2", "gen_words", "copy_words", "hit_cap", "loop"]
X = np.array([[d[f] for f in F] for d in R]); X[:, 7:9] = np.log1p(X[:, 7:9])
g_, c_ = np.array([d["rl_gen"] for d in R]), np.array([d["rl_copy"] for d in R])
y = (g_ > c_).astype(int)
n = len(R)
rng = np.random.default_rng(0); B = rng.integers(0, n, (2000, n))
def ci(a, b):
    d = a - b; bd = d[B].mean(1)
    return "%+.4f [95%% CI %+.4f, %+.4f]" % (d.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5))

rep = {"n": n, "mean_rougeL": {"generated_final": g_.mean(), "copy_reranked": c_.mean(),
                                "copy_original": np.mean([d["rl_copy_orig"] for d in R]), "P1": np.mean([d["rl_p1"] for d in R])},
       "generated_wins": int(y.sum()), "oracle_max_gen_copy": np.maximum(g_, c_).mean(),
       "mean_gap_when_gen_wins": float((g_ - c_)[y == 1].mean()), "mean_gap_when_copy_wins": float((c_ - g_)[y == 0].mean())}

# where does generation win? quartiles of each feature
rep["by_feature_quartile"] = {}
for j, f in enumerate(F):
    if f in ("hit_cap", "loop", "ret_top1_same"):
        groups = [(v, X[:, j] == v) for v in (0.0, 1.0)]
    else:
        qs = np.quantile(X[:, j], [0.25, 0.5, 0.75])
        b = np.digitize(X[:, j], qs)
        groups = [("Q%d" % (k + 1), b == k) for k in range(4)]
    rep["by_feature_quartile"][f] = {str(k): {"n": int(m.sum()), "gen": round(g_[m].mean(), 4), "copy": round(c_[m].mean(), 4),
                                              "gen_win_rate": round(y[m].mean(), 3)} for k, m in groups if m.sum()}

# gate: logistic regression on (gen beats copy), out-of-fold predictions, repeated 10x10 CV; choose gen when p > 0.5
def gate_oof(cols, seed):
    p = np.zeros(n)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X[tr][:, cols], y[tr])
        p[te] = m.predict_proba(X[te][:, cols])[:, 1]
    return p
rep["gates"] = {}
for name, cols in {"all_features": list(range(len(F))), "gen_vs_copy_only": [F.index("gen_vs_copy")],
                   "reranker_conf_only": [F.index("rr_top"), F.index("rr_margin")]}.items():
    sel = [np.where(gate_oof(cols, s) > 0.5, g_, c_) for s in range(10)]
    choice = np.mean(sel, 0)                       # average over the 10 CV repetitions
    rep["gates"][name] = {"rougeL": choice.mean(), "vs_copy_reranked": ci(choice, c_), "vs_generated": ci(choice, g_),
                          "share_choosing_generated": float(np.mean([(s == g_) & (g_ != c_) for s in sel]))}
full = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X, y)
rep["gate_coefficients_standardised"] = dict(zip(F, np.round(full[-1].coef_[0], 3).tolist()))
rep["feature_means"] = {"gen_vs_copy": X[:, F.index("gen_vs_copy")].mean(), "gen_words_median": float(np.median([d["gen_words"] for d in R])),
                        "copy_words_median": float(np.median([d["copy_words"] for d in R])),
                        "ref_words_median": float(np.median([len(r["final_answer"].split()) for r in hold]))}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1, default=float)
print(json.dumps(rep, indent=1, default=lambda v: round(float(v), 4)))
