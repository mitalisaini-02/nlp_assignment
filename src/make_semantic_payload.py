"""Collect the notebook 09 hold-out answers (300 questions x 7 systems + references) into one payload for notebook 10
(semantic evaluation). Re-scores ROUGE-L and asserts it equals metrics.json, so the payload is exactly what was reported.
    python -I src/make_semantic_payload.py "inference results second run" C:/nlp_data/diag/semantic_payload.json
"""
import csv, gzip, json, os, random, sys
import numpy as np
from rouge_score import rouge_scorer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "inference results second run")
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/nlp_data/diag/semantic_payload.json"
NN_ORIG = "C:/Users/ARTH/Downloads/question_neighbours.json.gz"
NN_RR = "C:/nlp_data/kaggle_out/p7/question_neighbours_reranked.json.gz"

rows = list(csv.DictReader(open(os.path.join(PROJ, "data/processed/train_processed.csv"), encoding="utf-8")))
qid2row = {r["question_id"]: r for r in rows}
hold = random.Random(42).sample([r for r in rows if r["split"] == "test"], 300)      # identical to notebook 09
NNO = json.load(gzip.open(NN_ORIG, "rt", encoding="utf-8"))
NNR = json.load(gzip.open(NN_RR, "rt", encoding="utf-8"))
gen = {}
for f in ("gen_gpu0.jsonl", "gen_gpu1.jsonl"):
    for l in open(os.path.join(RUN, "gen", f), encoding="utf-8"):
        d = json.loads(l); gen[d["uid"]] = d
metrics = json.load(open(os.path.join(RUN, "metrics.json"), encoding="utf-8"))

nearest = lambda NN, q: qid2row[NN["test"][q][0][0]]["final_answer"]
GEN = {"B1 zero-shot base model": "base|hold", "B2 fine-tuned v1, no retrieval": "none_v1|hold",
       "P1 fine-tuned v1 + retrieval (Phase 4)": "nn_v1|hold", "P2 FINAL fine-tuned v2 + re-ranked retrieval": "nn_v2|hold",
       "P2 ablation: original retrieval order": "nn_v2|hold_orig"}
items = []
for r in hold:
    q = r["question_id"]
    s = {"B0a copy nearest answer (original order)": nearest(NNO, q), "B0a copy nearest answer (re-ranked)": nearest(NNR, q)}
    for name, pre in GEN.items():
        t = gen["%s|%s" % (pre, q)]["text"]
        s[name] = t if t.strip() else nearest(NNR, q)                          # same empty-answer rule as notebook 09
    # the question of the copied neighbour, to see when a copy answers a different question
    items.append({"qid": q, "question": r["question"], "reference": r["final_answer"], "systems": s,
                  "copied_question_reranked": qid2row[NNR["test"][q][0][0]]["question"],
                  "copied_question_original": qid2row[NNO["test"][q][0][0]]["question"]})

sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
for name in items[0]["systems"]:
    m = np.mean([sc.score(it["reference"], it["systems"][name])["rougeL"].fmeasure for it in items])
    assert abs(m - metrics["systems"][name]["rougeL"]) < 1e-9, (name, m, metrics["systems"][name]["rougeL"])
    print("%-50s ROUGE-L %.4f  (= metrics.json)" % (name, m))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"source": "notebook 09 run arth99/notebookf1c39cfa36, hold-out N=300 (seed 42)", "items": items},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", OUT, len(items), "items,", round(os.path.getsize(OUT) / 1e6, 1), "MB")
