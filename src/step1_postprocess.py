"""Step 1 (CPU only): post-process the saved Phase 5 generations and re-score them on the same 300 hold-out rows.

Inputs (all saved files, no model):
  data/processed/train_processed.csv, question_neighbours.json.gz, kaggle_out/p5/gen/gen_gpu*.jsonl, kaggle_out/p5/metrics.json
Outputs (OUT dir): step1_metrics.json, step1_per_item.csv

Run:  C:/nlp_data/venv/Scripts/python.exe -I src/step1_postprocess.py
"""
import csv
import gzip
import json
import os
import random
import re
import sys
import glob

import numpy as np
from rouge_score import rouge_scorer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_harness as H  # noqa: E402

PROJ = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(PROJ, "data", "processed", "train_processed.csv")
NN_PATH = os.environ.get("NN_PATH", "C:/Users/ARTH/Downloads/question_neighbours.json.gz")
P5 = os.environ.get("P5_DIR", "C:/nlp_data/kaggle_out/p5")
OUT = os.environ.get("OUT_DIR", "C:/nlp_data/p5b")
N_HOLDOUT = 300

csv.field_size_limit(10 ** 9)
os.makedirs(OUT, exist_ok=True)


def op(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


# ---------------------------------------------------------------- data (identical to notebook 05)
rows = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
qid2row = {r["question_id"]: r for r in rows}
NN = json.load(op(NN_PATH))
test_rows = [r for r in rows if r["split"] == "test"]
hold = random.Random(42).sample(test_rows, min(N_HOLDOUT, len(test_rows)))

gen = {}
for f in glob.glob(P5 + "/gen/gen_gpu*.jsonl"):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            gen[d["uid"]] = d

ids = [r["question_id"] for r in hold if all(("%s|hold|%s" % (a, r["question_id"])) in gen for a in ("nn", "none", "base"))]
refs = {r["question_id"]: r["final_answer"] for r in hold if r["question_id"] in set(ids)}


def nb_answers(split, q, k):
    return [qid2row[n[0]]["final_answer"] for n in NN[split].get(q, [])[:k]]


def nearest(split, q):
    a = nb_answers(split, q, 1)
    return a[0] if a else ""


# ---------------------------------------------------------------- loop trimming
SENT_END = re.compile(r"[.!?:)]\s*$|\n")


def trim_loop(text, n=8):
    """Cut at the first word where an n-gram (lower-cased words) repeats an earlier one, then back off to the end of
    the last complete sentence/line before the cut. Returns (text, was_cut)."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words = [text[a:b].lower().strip(".,;:*#-()") for a, b in spans]
    seen = set()
    cut = None
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        if g in seen:
            cut = i
            break
        seen.add(g)
    if cut is None:
        return text, False
    head = text[:spans[cut][0]].rstrip()
    # back off to the last sentence/line end so the answer does not stop mid-sentence
    k = max(head.rfind(". "), head.rfind(".\n"), head.rfind("\n"))
    if head.endswith((".", "!", "?")):
        k = len(head)
    if k > len(head) * 0.5:
        head = head[:k + 1].rstrip()
    return head, True


def trim_tail(text):
    """For answers cut at the token limit without a loop: drop the unfinished last sentence."""
    t = text.rstrip()
    if t.endswith((".", "!", "?")):
        return t
    k = max(t.rfind(". "), t.rfind(".\n"), t.rfind("\n"))
    return t[:k + 1].rstrip() if k > len(t) * 0.7 else t


# ---------------------------------------------------------------- systems
SC_L = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rl(a, b):
    return SC_L.score(a, b)["rougeL"].fmeasure


def gtext(ad, q):
    t = gen["%s|hold|%s" % (ad, q)]["text"]
    return t if t.strip() else nearest("test", q)


systems = {
    "B0a nearest-neighbour answer (no model)": {q: nearest("test", q) for q in ids},
    "PROPOSED raw (as in metrics.json)": {q: gtext("nn", q) for q in ids},
}
cut_counts = {}
for n in (8, 12, 16):
    p, c = {}, 0
    for q in ids:
        t, was = trim_loop(gtext("nn", q), n)
        if not was and gen["nn|hold|%s" % q]["hit_limit"]:
            t = trim_tail(t)
        c += was
        p[q] = t
    systems["PROPOSED loop-trimmed n=%d" % n] = p
    cut_counts[n] = c
    print("loop-trim n=%d: %d of %d answers cut" % (n, c, len(ids)), flush=True)

# same trimming for B2 (ablation stays consistent)
systems["B2 fine-tuned no retrieval, loop-trimmed n=8"] = {q: trim_loop(gtext("none", q), 8)[0] for q in ids}

# MBR / consensus selection among [trimmed generation, top-3 neighbour answers]
TRIM = systems["PROPOSED loop-trimmed n=8"]
mbr, mbr_pick = {}, {}
print("MBR selection ...", flush=True)
for j, q in enumerate(ids):
    cands = [TRIM[q]] + nb_answers("test", q, 3)
    m = len(cands)
    S = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            S[a, b] = S[b, a] = rl(cands[a], cands[b])
    score = S.sum(1) / (m - 1)
    k = int(score.argmax())
    mbr[q], mbr_pick[q] = cands[k], k
    if (j + 1) % 50 == 0:
        print("  %d/%d" % (j + 1, len(ids)), flush=True)
systems["MBR over [trimmed gen, top-3 neighbours]"] = mbr
pick_counts = {("generated" if k == 0 else "neighbour_%d" % k): sum(1 for v in mbr_pick.values() if v == k) for k in range(4)}
print("MBR picks:", pick_counts)

# ---------------------------------------------------------------- scoring
res = {n: H.evaluate(p, refs) for n, p in systems.items()}
cols = ["rouge1", "rouge2", "rougeL", "bleu", "len_ratio_mean", "fact_f1"]
print("\n%-48s " % ("system (n=%d)" % len(ids)) + " ".join("%8s" % c[:8] for c in cols))
for n, r in res.items():
    print("%-48s " % n + " ".join("%8.4f" % r[c] for c in cols))

per = {n: np.array([rl(refs[q], p[q]) for q in ids]) for n, p in systems.items()}
idx = np.random.default_rng(0).integers(0, len(ids), (2000, len(ids)))


def diff_ci(a, b):
    d = (per[a] - per[b])[idx].mean(axis=1)
    return "%+.4f  [95%% CI %+.4f, %+.4f]" % (per[a].mean() - per[b].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))


B0, RAW = "B0a nearest-neighbour answer (no model)", "PROPOSED raw (as in metrics.json)"
boot = {}
for n in systems:
    if n not in (B0,):
        boot["%s - B0a" % n] = diff_ci(n, B0)
    if n not in (B0, RAW) and n.startswith(("PROPOSED", "MBR")):
        boot["%s - raw" % n] = diff_ci(n, RAW)
print("\nROUGE-L paired bootstrap (2000 resamples):")
for k, v in boot.items():
    print("  %-70s %s" % (k, v))

# sanity: must reproduce metrics.json
M = json.load(open(P5 + "/metrics.json"))["systems"]
chk = {"B0a": (res[B0]["rougeL"], M["B0a nearest-neighbour answer (no model)"]["rougeL"]),
       "proposed": (res[RAW]["rougeL"], M["PROPOSED fine-tuned + retrieval"]["rougeL"])}
print("\nsanity vs metrics.json:", {k: "%.6f vs %.6f" % v for k, v in chk.items()})
ok = all(abs(a - b) < 1e-6 for a, b in chk.values())
print("REPRODUCED" if ok else "MISMATCH - do not trust the numbers above")

json.dump({"holdout_n": len(ids), "systems": res, "bootstrap_rougeL": boot, "loop_cut_counts": cut_counts,
           "mbr_picks": pick_counts, "sanity_reproduced": ok, "source_generations": P5 + "/gen"},
          open(OUT + "/step1_metrics.json", "w"), indent=1, default=float)
with open(OUT + "/step1_per_item.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    names = list(systems)
    w.writerow(["question_id", "ref_words", "hit_limit"] + ["rougeL|" + n for n in names] + ["mbr_pick"])
    for i, q in enumerate(ids):
        w.writerow([q, len(refs[q].split()), gen["nn|hold|%s" % q]["hit_limit"]] + ["%.4f" % per[n][i] for n in names] + [mbr_pick[q]])
print("wrote", OUT + "/step1_metrics.json")
