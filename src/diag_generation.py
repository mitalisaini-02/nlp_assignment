"""Diagnostics B + C (CPU, saved files only).

B. Question-retrieval headroom on the same 300 hold-out rows as Phase 5: ROUGE-L of each of the top-10 neighbour
   answers vs the reference; oracle@k; does a better reference answer go with a better generated answer?
C. Repetition loops: rate per system, where loops start, what they correlate with (reference length, prompt length,
   neighbour quality), and whether the human training answers themselves contain loop-like repetition.

Run:  C:/nlp_data/venv/Scripts/python.exe -I src/diag_generation.py
"""
import csv
import glob
import gzip
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(PROJ, "data", "processed", "train_processed.csv")
NN_PATH = "C:/Users/ARTH/Downloads/question_neighbours.json.gz"
P5, P4 = "C:/nlp_data/kaggle_out/p5", "C:/nlp_data/kaggle_out/p4"
OUT = "C:/nlp_data/diag"
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10 ** 9)
t0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
qid2row = {r["question_id"]: r for r in rows}
NN = json.load(gzip.open(NN_PATH, "rt", encoding="utf-8"))
test_rows = [r for r in rows if r["split"] == "test"]
hold = random.Random(42).sample(test_rows, 300)
gen = {}
for f in glob.glob(P5 + "/gen/gen_gpu*.jsonl"):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            gen[d["uid"]] = d
jobs = {}
for line in open(P5 + "/jobs.jsonl", encoding="utf-8"):
    d = json.loads(line)
    jobs[d["uid"]] = d
ids = [r["question_id"] for r in hold]
refs = {r["question_id"]: r["final_answer"] for r in hold}
SC = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rl(ref, pred):
    return SC.score(ref, pred)["rougeL"].fmeasure


# ================================================================ C. loops
WORD = re.compile(r"\S+")


def loop_info(text, n=8, min_rep=3):
    """A loop = some n-gram occurs >= min_rep times. Returns (is_loop, start_word, n_words)."""
    w = [x.lower().strip(".,;:*#-()") for x in WORD.findall(text)]
    cnt = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    rep = {g for g, c in cnt.items() if c >= min_rep}
    if not rep:
        return False, None, len(w)
    start = next(i for i in range(len(w) - n + 1) if tuple(w[i:i + n]) in rep)
    return True, start, len(w)


res = {"loops": {}, "neighbour_headroom": {}, "training_data": {}}
print("\nC1. LOOPS PER SYSTEM (loop = an 8-gram occurring >= 3 times)")
for name, ad, tag in (("PROPOSED nn hold", "nn", "hold"), ("B2 none hold", "none", "hold"), ("B1 base hold", "base", "hold"),
                      ("nn unseen xt", "nn", "xt")):
    G = [d for u, d in gen.items() if d["adapter"] == ad and u.split("|")[1] == tag]
    L = [loop_info(d["text"]) for d in G]
    hit = [d["hit_limit"] for d in G]
    lp = [x[0] for x in L]
    starts = [x[1] for x in L if x[0]]
    o = {"n": len(G), "hit_limit": int(sum(hit)), "loops": int(sum(lp)), "loops_and_hit_limit": int(sum(a and b for a, b in zip(lp, hit))),
         "loop_start_word_median": float(np.median(starts)) if starts else None,
         "loop_start_word_p25_p75": [float(np.percentile(starts, 25)), float(np.percentile(starts, 75))] if starts else None,
         "tokens_per_word": float(sum(d["n_tokens"] for d in G) / max(1, sum(x[2] for x in L)))}
    res["loops"][name] = o
    print("  %-18s n=%d  hit_limit=%d  loops=%d (of which hit limit %d)  loop starts at word median %s IQR %s  tok/word %.2f" % (
        name, o["n"], o["hit_limit"], o["loops"], o["loops_and_hit_limit"], o["loop_start_word_median"], o["loop_start_word_p25_p75"], o["tokens_per_word"]))

# per-item table for the proposed model on the hold-out
tpw = res["loops"]["PROPOSED nn hold"]["tokens_per_word"]
items = []
for q in ids:
    d = gen["nn|hold|%s" % q]
    lp, st, nw = loop_info(d["text"])
    nb = NN["test"].get(q, [])
    items.append({"q": q, "loop": lp, "hit": d["hit_limit"], "ref_words": len(refs[q].split()),
                  "prompt_words": len(jobs["nn|hold|%s" % q]["user"].split()),
                  "nb1_words": len(qid2row[nb[0][0]]["final_answer"].split()) if nb else 0,
                  "nb1_sim": nb[0][1] if nb else 0.0,
                  "nb_loop": any(loop_info(qid2row[n[0]]["final_answer"])[0] for n in nb[:2])})


def by_bins(key, edges, label):
    print("  loop rate by %s:" % label)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = [it for it in items if lo <= it[key] < hi]
        if s:
            r = sum(it["loop"] for it in s) / len(s)
            out.append({"bin": [lo, hi], "n": len(s), "loop_rate": r, "hit_rate": sum(it["hit"] for it in s) / len(s)})
            print("     [%6s, %6s)  n=%3d  loop %.2f  hit-limit %.2f" % (lo, hi, len(s), r, out[-1]["hit_rate"]))
    res["loops"]["by_" + key] = out


print("\nC2. WHAT LOOPS CORRELATE WITH (proposed, 300 hold-out)")
by_bins("ref_words", [0, 250, 400, 600, 900, 99999], "reference length (words)")
by_bins("prompt_words", [0, 500, 650, 800, 99999], "prompt length (words)")
by_bins("nb1_words", [0, 250, 400, 600, 99999], "nearest neighbour answer length (words)")
by_bins("nb1_sim", [0, 0.08, 0.1, 0.12, 1], "nearest neighbour retrieval score")
nl = [it for it in items if it["nb_loop"]]
print("  items whose top-2 neighbour answers contain loop-like repetition: %d, loop rate there %.2f vs %.2f elsewhere" % (
    len(nl), np.mean([it["loop"] for it in nl]) if nl else 0, np.mean([it["loop"] for it in items if not it["nb_loop"]])))
print("  reference length that the %d-token cap allows: ~%d words (tok/word %.2f); references longer than that: %d of 300" % (
    768, 768 / tpw, tpw, sum(it["ref_words"] > 768 / tpw for it in items)))
res["loops"]["refs_longer_than_cap"] = int(sum(it["ref_words"] > 768 / tpw for it in items))

# examples of the looping segment
print("\n  three loop examples (text around the loop start):")
shown = 0
for q in ids:
    d = gen["nn|hold|%s" % q]
    lp, st, _ = loop_info(d["text"])
    if lp and shown < 3:
        w = WORD.findall(d["text"])
        print("   Q: %s\n     ...%s ||LOOP|| %s..." % (qid2row[q]["question"][:90], " ".join(w[max(0, st - 25):st]), " ".join(w[st:st + 45])))
        shown += 1

# C3. the human answers themselves
print("\nC3. TRAINING TARGETS")
tr = [json.loads(line) for line in open(P4 + "/data/train_nn.jsonl", encoding="utf-8")]
tl = [loop_info(d["answer"]) for d in tr]
allr = [loop_info(r["final_answer"]) for r in rows]
aw = np.array([len(d["answer"].split()) for d in tr])
res["training_data"] = {"train_targets_n": len(tr), "train_targets_with_loop_like_repetition": int(sum(x[0] for x in tl)),
                        "all_15000_answers_with_loop_like_repetition": int(sum(x[0] for x in allr)),
                        "train_target_words_median_p90": [float(np.median(aw)), float(np.percentile(aw, 90))],
                        "train_targets_longer_than_cap_words": int((aw > 768 / tpw).sum())}
print("  ", res["training_data"])

# ================================================================ B. neighbour headroom
print("\nB. QUESTION-RETRIEVAL HEADROOM (300 hold-out rows, full texts)")
K = 10
M = np.full((len(ids), K), np.nan)
prop_rl = np.zeros(len(ids))
for i, q in enumerate(ids):
    nb = NN["test"].get(q, [])[:K]
    for j, n in enumerate(nb):
        M[i, j] = rl(refs[q], qid2row[n[0]]["final_answer"])
    t = gen["nn|hold|%s" % q]["text"]
    prop_rl[i] = rl(refs[q], t if t.strip() else qid2row[nb[0][0]]["final_answer"])
    if (i + 1) % 50 == 0:
        log("  scored %d/%d" % (i + 1, len(ids)))
Mz = np.nan_to_num(M, nan=0.0)
by_rank = [float(np.nanmean(M[:, j])) for j in range(K)]
oracle = {k: float(Mz[:, :k].max(1).mean()) for k in (1, 2, 3, 5, 10)}
best_rank = Counter(int(x) for x in Mz.argmax(1))
sims = np.array([NN["test"][q][0][1] for q in ids])
corr_sim = float(np.corrcoef(sims, Mz[:, 0])[0, 1])
corr_ref_gen = float(np.corrcoef(Mz[:, 0], prop_rl)[0, 1])
q4 = np.quantile(Mz[:, 0], [0.25, 0.5, 0.75])
bins = np.digitize(Mz[:, 0], q4)
gen_by_nbq = [{"nb1_quartile": b + 1, "n": int((bins == b).sum()), "nb1_rougeL": float(Mz[bins == b, 0].mean()),
               "proposed_rougeL": float(prop_rl[bins == b].mean())} for b in range(4)]
res["neighbour_headroom"] = {"n": len(ids), "mean_rougeL_by_rank": by_rank, "oracle_at_k": oracle,
                             "rank_of_best_neighbour_counts": dict(sorted(best_rank.items())),
                             "corr_retrieval_score_vs_nb1_rougeL": corr_sim, "corr_nb1_rougeL_vs_proposed_rougeL": corr_ref_gen,
                             "proposed_by_nb1_quality_quartile": gen_by_nbq, "proposed_mean_rougeL": float(prop_rl.mean())}
print("  mean ROUGE-L by neighbour rank 1..10:", " ".join("%.3f" % x for x in by_rank))
print("  oracle best-of-k:", {k: round(v, 4) for k, v in oracle.items()})
print("  rank of the best neighbour (0 = top-1):", dict(sorted(best_rank.items())))
print("  corr(retrieval score, top-1 answer ROUGE-L) = %.3f" % corr_sim)
print("  corr(top-1 neighbour ROUGE-L, generated ROUGE-L) = %.3f" % corr_ref_gen)
for g in gen_by_nbq:
    print("   neighbour-quality quartile %d (n=%d): top-1 neighbour RL %.3f -> generated RL %.3f" % (
        g["nb1_quartile"], g["n"], g["nb1_rougeL"], g["proposed_rougeL"]))

json.dump(res, open(OUT + "/diag_generation.json", "w"), indent=1, default=float)
log("wrote", OUT + "/diag_generation.json")
