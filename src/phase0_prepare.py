"""Phase 0: data foundation (local, CPU).

1. Normalise questions; repeated questions form one group.
2. Per group, pick the medoid answer (highest mean ROUGE-L F1 against the rest of the group).
3. Split 90/10 by GROUP (seed fixed) -> no question text appears in both train and test.
   A 5% dev set (of the 90%) is carved out for tuning. IDs saved to data/processed/split.json.
4. Parse source citations per row; normalise the unseen set and flag its overlap with train.
5. Leave-one-out estimate of how well "return the stored answer" does for repeated questions
   (the only honest way to size the exact-match router, since the grouped split has no overlap).

Run:  C:\\nlp_data\\venv\\Scripts\\python.exe src\\phase0_prepare.py
"""
import csv
import json
import os
import random
import re
import statistics as st
import sys
import unicodedata
from collections import defaultdict

from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(__file__))
from sources import parse_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NLP_DATA_DIR", ROOT)  # where the two raw CSVs live
OUT = os.environ.get("NLP_OUT_DIR", os.path.join(ROOT, "data", "processed"))
SEED = 42
TEST_FRAC = 0.10
DEV_FRAC_OF_TRAIN = 0.05
csv.field_size_limit(10 ** 9)


def norm_question(q):
    """NFKC + lowercase + keep letters/digits/combining marks (so Indic scripts survive) + collapse spaces."""
    q = unicodedata.normalize("NFKC", q or "").lower()
    q = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in q)
    return re.sub(r"\s+", " ", q).strip()


def read(name):
    return list(csv.DictReader(open(os.path.join(DATA_DIR, name), encoding="utf-8")))


def main():
    os.makedirs(OUT, exist_ok=True)
    train = read("training_set_15000.csv")
    test = read("unseen_set_1000.csv")
    for r in train:
        r["norm_q"] = norm_question(r["question"])
    for r in test:
        r["norm_q"] = norm_question(r["question"])

    # ---- 1. groups
    gid_of, members = {}, defaultdict(list)
    for i, r in enumerate(train):
        g = gid_of.setdefault(r["norm_q"], "g%05d" % len(gid_of))
        r["group_id"] = g
        members[g].append(i)
    print("rows", len(train), "groups", len(members), "repeated groups", sum(len(v) > 1 for v in members.values()))

    # ---- 2. medoids + pairwise ROUGE-L (also feeds the leave-one-out estimate)
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    medoid_idx, loo_scores, loo_random = {}, [], []
    for g, idx in members.items():
        if len(idx) == 1:
            medoid_idx[g] = idx[0]
            continue
        n = len(idx)
        M = [[0.0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                f = sc.score(train[idx[a]]["final_answer"], train[idx[b]]["final_answer"])["rougeL"].fmeasure
                M[a][b] = M[b][a] = f
        mean = [sum(M[a]) / (n - 1) for a in range(n)]
        # tie-break (e.g. size-2 groups are symmetric): lowest final_answer_id, deterministic
        best = max(range(n), key=lambda a: (round(mean[a], 9), -int(train[idx[a]]["final_answer_id"], 16)))
        medoid_idx[g] = idx[best]
        # leave-one-out: hold member h out, predict medoid of the remaining members, score vs h
        for h in range(n):
            rest = [a for a in range(n) if a != h]
            if len(rest) == 1:
                pred = rest[0]
            else:
                pred = max(rest, key=lambda a: (round(sum(M[a][b] for b in rest if b != a) / (len(rest) - 1), 9),
                                                -int(train[idx[a]]["final_answer_id"], 16)))
            loo_scores.append(M[h][pred])
            loo_random.append(st.mean(M[h][a] for a in rest))
    for g, i in medoid_idx.items():
        train[i]["is_medoid"] = 1
    for r in train:
        r.setdefault("is_medoid", 0)

    # ---- 3. grouped split
    rng = random.Random(SEED)
    groups = sorted(members)
    rng.shuffle(groups)
    n_rows = len(train)
    test_target = round(TEST_FRAC * n_rows)
    split_of, counts = {}, defaultdict(int)
    for g in groups:
        if counts["test"] < test_target:
            s = "test"
        else:
            s = "train_pool"
        split_of[g] = s
        counts[s] += len(members[g])
    dev_target = round(DEV_FRAC_OF_TRAIN * counts["train_pool"])
    counts["dev"] = 0
    for g in groups:
        if split_of[g] == "train_pool":
            if counts["dev"] < dev_target:
                split_of[g] = "dev"
                counts["dev"] += len(members[g])
            else:
                split_of[g] = "train"
    for r in train:
        r["split"] = split_of[r["group_id"]]
    sizes = {s: sum(1 for r in train if r["split"] == s) for s in ("train", "dev", "test")}
    print("split rows", sizes)
    assert not ({r["norm_q"] for r in train if r["split"] == "test"} &
                {r["norm_q"] for r in train if r["split"] != "test"}), "question leaked across split"

    # ---- 4. citations, unseen set overlap
    for r in train:
        r["n_words_answer"] = len(r["final_answer"].split())
        r["group_size"] = len(members[r["group_id"]])
        cits = [c for c in parse_sources(r["final_answer_sources"]) if c["kind"] in ("zoho", "external")]
        r["citations_json"] = json.dumps(
            [{"doc_id": c["file_id"] or c["url"], "kind": c["kind"], "pages": c["pages"], "name": c["name"],
              "source_type": c["source_type"]} for c in cits], ensure_ascii=False)
    cols = ["question_id", "question", "norm_q", "group_id", "group_size", "is_medoid", "split",
            "final_answer_id", "final_answer", "n_words_answer", "citations_json"]
    with open(os.path.join(OUT, "train_processed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(train)

    in_train = 0
    for r in test:
        g = gid_of.get(r["norm_q"])
        r["in_train"] = int(g is not None)
        r["matched_group_id"] = g or ""
        r["matched_group_size"] = len(members[g]) if g else 0
        in_train += r["in_train"]
    with open(os.path.join(OUT, "test_processed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "question", "norm_q", "in_train", "matched_group_id",
                                          "matched_group_size"])
        w.writeheader()
        w.writerows(test)

    json.dump({
        "seed": SEED, "test_frac": TEST_FRAC, "dev_frac_of_train": DEV_FRAC_OF_TRAIN, "rows": sizes,
        "ids": {s: [r["question_id"] for r in train if r["split"] == s] for s in ("train", "dev", "test")},
    }, open(os.path.join(OUT, "split.json"), "w"))

    # ---- 5. report
    wl = [r["n_words_answer"] for r in train]
    rep = {
        "train_rows": n_rows, "groups": len(members),
        "repeated_groups": sum(len(v) > 1 for v in members.values()),
        "split_rows": sizes,
        "unseen_rows": len(test), "unseen_exact_in_train": in_train,
        "answer_words_median": st.median(wl),
        "loo_medoid_rougeL_mean": round(st.mean(loo_scores), 4),
        "loo_random_other_rougeL_mean": round(st.mean(loo_random), 4),
        "loo_pairs": len(loo_scores),
    }
    json.dump(rep, open(os.path.join(OUT, "phase0_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
