"""Build notebooks/10_semantic_evaluation.ipynb (BERTScore + bge-m3 similarity of the notebook 09 hold-out answers).
The notebook reads the saved answers from the notebook 09 output (Kaggle limits notebook source to 1 MB, so nothing is embedded)
and asserts that its ROUGE-L equals notebook 09's metrics.json.
    python -I src/build_nb10.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebooks import ROOT, code, md, notebook

def build():
    cells = [
        md("""
# 10 - Semantic evaluation: does the ranking change with a meaning-based metric?

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU** (one T4 is enough), **Internet = On**.
> Nothing is generated here: this notebook only scores the answers saved by the final evaluation (notebook 09). About 15-20 minutes.
>
> **Inputs (Add Input):**
> 1. `train_processed.csv` (dataset `train-processed1`) - questions and human reference answers
> 2. `question_neighbours.json` (dataset `nlp-qna-retrieval`) - original retrieval order
> 3. `question_neighbours_reranked.json.gz` (Your Work -> re-ranker notebook 07 output)
> 4. the final evaluation output (Your Work -> notebook 09 output: `gen/`, `metrics.json`)

**Why.** On ROUGE-L / BLEU (word overlap) copying the top re-ranked training answer beats the fine-tuned generator. Word overlap rewards
re-using human wording; it does not check whether an answer means the same thing or addresses the question that was asked. This notebook
scores the same answers with meaning-based measures:

| measure | what it compares | notes |
|---|---|---|
| **BERTScore F1** (roberta-large, layer 17 = the library default for English) | answer vs human reference, token embeddings matched by meaning | standard metric; the model reads at most 512 tokens, so long answers are cut (the share is printed) |
| BERTScore F1, rescaled | same, on the library's baseline-rescaled scale | a linear transform: same ranking, easier to read |
| **answer-reference similarity** (bge-m3 cosine) | whole answer vs whole reference (up to 2,048 tokens) | covers long answers fully |
| **question-answer relevance** (bge-m3 cosine) | the asked question vs the answer, **no reference used** | does the answer address this question? Caveat: restating the question raises it |

Every comparison uses the same N = 300 questions, with paired-bootstrap 95% confidence intervals (2,000 resamples).
"""),
        code("!pip install -q bert-score rouge-score sentence-transformers"),
        md("## 0. Load the saved answers (300 hold-out questions x 7 systems + references)"),
        code('''
import glob, os, json, gzip, csv, random
import numpy as np
from rouge_score import rouge_scorer
csv.field_size_limit(10 ** 9)
INPUT = os.environ.get("P10_INPUT", "/kaggle/input")
def find(pattern):
    hit = sorted(p.replace("\\\\", "/") for p in glob.glob(INPUT + "/**/" + pattern, recursive=True))
    assert hit, "attach a dataset / notebook output containing " + pattern
    return hit[0]
op = lambda p: gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
GEN0 = find("gen/gen_gpu0.jsonl")
RUN = os.path.dirname(os.path.dirname(GEN0))                     # the notebook 09 output folder
rows = list(csv.DictReader(op(find("train_processed*.csv"))))
NNO, NNR = json.load(op(find("question_neighbours.json*"))), json.load(op(find("question_neighbours_reranked.json*")))
metrics09 = json.load(open(os.path.join(RUN, "metrics.json")))
gen = {}
for f in glob.glob(os.path.join(RUN, "gen", "gen_gpu*.jsonl")):
    for l in open(f, encoding="utf-8"):
        d = json.loads(l); gen[d["uid"]] = d
qid2row = {r["question_id"]: r for r in rows}
hold = random.Random(42).sample([r for r in rows if r["split"] == "test"], metrics09["holdout_n"])   # same sample as notebook 09
nearest = lambda NN, q: qid2row[NN["test"][q][0][0]]["final_answer"]
GEN_SYS = {"B1 zero-shot base model": "base|hold", "B2 fine-tuned v1, no retrieval": "none_v1|hold",
           "P1 fine-tuned v1 + retrieval (Phase 4)": "nn_v1|hold", "P2 FINAL fine-tuned v2 + re-ranked retrieval": "nn_v2|hold",
           "P2 ablation: original retrieval order": "nn_v2|hold_orig"}
items = []
for r in hold:
    q = r["question_id"]
    s = {"B0a copy nearest answer (original order)": nearest(NNO, q), "B0a copy nearest answer (re-ranked)": nearest(NNR, q)}
    for name, pre in GEN_SYS.items():
        t = gen["%s|%s" % (pre, q)]["text"]
        s[name] = t if t.strip() else nearest(NNR, q)               # same empty-answer rule as notebook 09
    items.append({"qid": q, "question": r["question"], "reference": r["final_answer"], "systems": s,
                  "copied_question_reranked": qid2row[NNR["test"][q][0][0]]["question"]})
_rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
for name in items[0]["systems"]:                                  # proves these are exactly the answers notebook 09 reported
    m = np.mean([_rs.score(it["reference"], it["systems"][name])["rougeL"].fmeasure for it in items])
    assert abs(m - metrics09["systems"][name]["rougeL"]) < 1e-9, (name, m)
    print("%-50s ROUGE-L %.4f = notebook 09 metrics.json" % (name, m))
payload = {"source": "notebook 09 output " + RUN + ", hold-out N=%d (seed 42)" % len(items), "items": items}
'''),
        md("## 1. Configuration"),
        code('''
import os, json, time, numpy as np, pandas as pd
import torch
CPU = os.environ.get("P10_CPU") == "1"                     # local smoke test only
WORK = os.environ.get("P10_WORK", "/kaggle/working")
N = int(os.environ.get("P10_N", "0"))                      # 0 = all 300
BERT_MODEL, BERT_LAYERS = "roberta-large", 17              # bert-score's default model/layer for English
EMB_MODEL, EMB_MAX_TOKENS = "BAAI/bge-m3", 2048
if not CPU:
    assert torch.cuda.is_available(), "No GPU. Right panel -> Session options -> Accelerator -> GPU, then Run All."
DEV = "cuda" if torch.cuda.is_available() else "cpu"
items = payload["items"][:N] if N else payload["items"]
SYS = list(items[0]["systems"])
B0O, B0R = "B0a copy nearest answer (original order)", "B0a copy nearest answer (re-ranked)"
FINAL, P1 = "P2 FINAL fine-tuned v2 + re-ranked retrieval", "P1 fine-tuned v1 + retrieval (Phase 4)"
B1, B2, ABL = "B1 zero-shot base model", "B2 fine-tuned v1, no retrieval", "P2 ablation: original retrieval order"
refs = [it["reference"] for it in items]
hyp = {s: [it["systems"][s] for it in items] for s in SYS}
per = {}                                                   # per[metric][system] = array over the questions
print("questions:", len(items), "| systems:", len(SYS), "| device:", DEV)
'''),
        md("## 2. BERTScore"),
        code('''
from bert_score import BERTScorer
t0 = time.time()
scorer = BERTScorer(model_type=BERT_MODEL, num_layers=BERT_LAYERS, lang="en", rescale_with_baseline=True,
                    device=DEV, batch_size=16)
base_f = float(scorer.baseline_vals[2])                    # rescaled = (raw - base) / (1 - base)
tk = scorer._tokenizer
ntok = lambda t: len(tk(t, add_special_tokens=False)["input_ids"])
print("references longer than 510 tokens (BERTScore sees only their start): %.0f%%" % (100 * np.mean([ntok(r) > 510 for r in refs])))
per["bertscore_f1"], per["bertscore_f1_rescaled"] = {}, {}
for s in SYS:
    _, _, F = scorer.score(hyp[s], refs)
    F = F.cpu().numpy().astype(float)
    per["bertscore_f1_rescaled"][s] = F
    per["bertscore_f1"][s] = F * (1 - base_f) + base_f
    print("  %-48s BERTScore F1 %.4f" % (s, per["bertscore_f1"][s].mean()))
del scorer; torch.cuda.empty_cache() if DEV == "cuda" else None
print("BERTScore done in %.1f min" % ((time.time() - t0) / 60))
'''),
        md("## 3. Whole-answer similarity and question relevance (bge-m3)"),
        code('''
from sentence_transformers import SentenceTransformer
t0 = time.time()
emb = SentenceTransformer(EMB_MODEL, device=DEV)
emb.max_seq_length = EMB_MAX_TOKENS
if DEV == "cuda":
    emb.half()
enc = lambda texts: emb.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
E_ref = enc(refs)
E_q = enc([it["question"] for it in items])
E_cq_rr = enc([it["copied_question_reranked"] for it in items])     # the training question whose answer B0a (re-ranked) copies
per["cos_answer_reference"], per["cos_question_answer"] = {}, {}
for s in SYS:
    E = enc(hyp[s])
    per["cos_answer_reference"][s] = (E * E_ref).sum(1).astype(float)
    per["cos_question_answer"][s] = (E * E_q).sum(1).astype(float)
    print("  %-48s answer-ref %.4f | question-answer %.4f" % (s, per["cos_answer_reference"][s].mean(), per["cos_question_answer"][s].mean()))
human_q_relevance = (E_ref * E_q).sum(1).astype(float)              # yardstick: how on-question the human answers are
asked_vs_copied_q = (E_q * E_cq_rr).sum(1).astype(float)
print("human reference question-answer relevance: %.4f" % human_q_relevance.mean())
print("bge-m3 done in %.1f min" % ((time.time() - t0) / 60))
'''),
        md("## 4. Results table (with ROUGE-L for comparison) and paired bootstraps"),
        code('''
from rouge_score import rouge_scorer
rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
per["rougeL"] = {s: np.array([rs.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyp[s])]) for s in SYS}
METRICS = ["rougeL", "bertscore_f1", "bertscore_f1_rescaled", "cos_answer_reference", "cos_question_answer"]
table = pd.DataFrame({s: {m: per[m][s].mean() for m in METRICS} for s in SYS}).T
table.loc["(human reference answer)", "cos_question_answer"] = human_q_relevance.mean()
pd.set_option("display.width", 250)
print("N = %d hold-out questions" % len(items))
print(table.round(4).to_string())

n = len(items)
idx = np.random.default_rng(0).integers(0, n, (2000, n))
def diff_ci(m, a, b):
    d = per[m][a] - per[m][b]
    bd = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
            "a_wins": int((d > 0).sum()), "b_wins": int((d < 0).sum())}
PAIRS = [(FINAL, B0R, "final - copy re-ranked nearest answer"), (FINAL, B0O, "final - copy nearest answer (original order)"),
         (FINAL, P1, "final - Phase 4 model"), (FINAL, B2, "final - no retrieval"), (FINAL, B1, "final - zero-shot"),
         (FINAL, ABL, "final - same model, original order"), (B0R, B0O, "re-ranked copy - original copy"),
         (P1, B0O, "Phase 4 model - original copy")]
boot = {m: {lab: diff_ci(m, a, b) for a, b, lab in PAIRS} for m in METRICS}
for m in METRICS:
    print("\\n%s  (paired bootstrap, n=%d)" % (m, n))
    for lab, r in boot[m].items():
        flag = "  *" if r["lo"] > 0 or r["hi"] < 0 else ""
        print("  %-48s %+.4f  [95%% CI %+.4f, %+.4f]  wins %d vs %d%s" % (lab, r["diff"], r["lo"], r["hi"], r["a_wins"], r["b_wins"], flag))
print("\\n* = the confidence interval excludes 0")
'''),
        md("""
## 5. When the copy answers a different question

The re-ranked copy is the answer to a *neighbouring* training question. Below, the questions are split into quartiles by how similar that
neighbouring question is to the asked one (bge-m3 cosine). If copying fails on meaning, it should fail where the neighbour is least similar.
"""),
        code('''
qs = np.quantile(asked_vs_copied_q, [0.25, 0.5, 0.75])
bins = np.digitize(asked_vs_copied_q, qs)
by_q = {}
for k in range(4):
    msk = bins == k
    row = {"n": int(msk.sum()), "asked_vs_copied_question_cos": float(asked_vs_copied_q[msk].mean())}
    for m in METRICS:
        row[m + "|final"] = float(per[m][FINAL][msk].mean())
        row[m + "|copy"] = float(per[m][B0R][msk].mean())
        row[m + "|final_win_rate"] = float((per[m][FINAL][msk] > per[m][B0R][msk]).mean())
    by_q["Q%d" % (k + 1)] = row
print(pd.DataFrame(by_q).round(4).to_string())
print("\\nLeast similar copied questions (asked -> copied):")
for i in np.argsort(asked_vs_copied_q)[:5]:
    print("  %.3f | %s\\n        -> %s" % (asked_vs_copied_q[i], items[i]["question"][:140], items[i]["copied_question_reranked"][:140]))
'''),
        md("## 6. Save"),
        code('''
out = {"n": len(items), "source": payload["source"], "models": {"bertscore": "%s layer %d (rescaled baseline F %.4f)" % (BERT_MODEL, BERT_LAYERS, base_f),
       "embedding": "%s, max %d tokens" % (EMB_MODEL, EMB_MAX_TOKENS)},
       "means": {s: {m: float(per[m][s].mean()) for m in METRICS} for s in SYS},
       "human_reference_question_relevance": float(human_q_relevance.mean()),
       "bootstrap": boot, "by_copied_question_similarity": by_q}
json.dump(out, open(os.path.join(WORK, "semantic_metrics.json"), "w"), indent=1)
flat = pd.DataFrame({"question_id": [it["qid"] for it in items], "asked_vs_copied_question_cos": asked_vs_copied_q,
                     "human_question_relevance": human_q_relevance})
for m in METRICS:
    for s in SYS:
        flat["%s|%s" % (m, s)] = per[m][s]
flat.to_csv(os.path.join(WORK, "per_item_semantic.csv"), index=False)
print("wrote semantic_metrics.json and per_item_semantic.csv")
'''),
        md("""
## After it finishes
Send me the notebook link (or the downloaded output folder). Output: `semantic_metrics.json` (all means, bootstraps and the quartile
analysis) and `per_item_semantic.csv` (every score for every question and system).
"""),
    ]
    return notebook(cells)


if __name__ == "__main__":
    p = os.path.join(ROOT, "notebooks", "10_semantic_evaluation.ipynb")
    json.dump(build(), open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p, round(os.path.getsize(p) / 1e6, 2), "MB")
