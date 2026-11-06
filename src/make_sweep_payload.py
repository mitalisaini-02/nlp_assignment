"""Builds the decoding-sweep payload: the first N of the Phase 5 hold-out sample (same seed-42 sample of 300), with the
exact Phase 5 prompts (jobs.jsonl), the reference answer, the B0a answer and the saved Phase 5 generation.

Run:  C:/nlp_data/venv/Scripts/python.exe -I src/make_sweep_payload.py   -> C:/nlp_data/diag/sweep_payload.json
"""
import csv
import glob
import gzip
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(os.path.dirname(HERE), "data", "processed", "train_processed.csv")
NN_PATH = "C:/Users/ARTH/Downloads/question_neighbours.json.gz"
P5 = "C:/nlp_data/kaggle_out/p5"
OUT = "C:/nlp_data/diag/sweep_payload.json"
N = 50
csv.field_size_limit(10 ** 9)

rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
qid2row = {r["question_id"]: r for r in rows}
NN = json.load(gzip.open(NN_PATH, "rt", encoding="utf-8"))
hold = random.Random(42).sample([r for r in rows if r["split"] == "test"], 300)[:N]
jobs = {}
for line in open(P5 + "/jobs.jsonl", encoding="utf-8"):
    d = json.loads(line)
    jobs[d["uid"]] = d
gen = {}
for f in glob.glob(P5 + "/gen/gen_gpu*.jsonl"):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            gen[d["uid"]] = d
items = []
for r in hold:
    q = r["question_id"]
    j, g = jobs["nn|hold|%s" % q], gen["nn|hold|%s" % q]
    items.append({"qid": q, "question": r["question"], "system": j["system"], "user": j["user"], "reference": r["final_answer"],
                  "b0a": qid2row[NN["test"][q][0][0]]["final_answer"], "phase5_text": g["text"], "phase5_hit_limit": g["hit_limit"]})
json.dump({"source": "Phase 5 hold-out sample (seed 42), first %d of 300; prompts from jobs.jsonl (adapter nn)" % N, "items": items},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", OUT, "| items", len(items), "| phase5 answers that hit the 768 cap:", sum(i["phase5_hit_limit"] for i in items),
      "| size %.0f KB" % (os.path.getsize(OUT) / 1024))
