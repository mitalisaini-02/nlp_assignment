"""Build notebooks/11_final_inference_full_test.ipynb: the submission inference notebook.
Loads the trained checkpoints, answers ALL 1,500 test questions with every system and the 1,000 unseen questions (Xt),
scores everything (lexical + BERTScore + bge-m3), writes the prediction CSVs. Resumable across Kaggle sessions, and reuses
answers from notebook 09 only when checkpoint hash, decoding and prompt are identical.
    python -I src/build_nb11.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebooks import ROOT, code, md, notebook


def src(name):
    return open(os.path.join(ROOT, "src", name), encoding="utf-8").read()


def build():
    cells = [
        md("""
# 11 - Final inference: all systems on the full test set + the unseen questions (Xt)

> Run with **Save Version -> Save & Run All (Commit)**. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
> This is the **inference notebook** of the submission. It loads the trained checkpoints (no training happens here) and generates
> answers for **all 1,500 test questions** (our 10% hold-out split, never seen in training) with every system, and for the
> **1,000 unseen questions (Xt)** with the final system. It then scores every system.

**Inputs (Add Input):**
1. `train_processed.csv` (dataset `train-processed1`) and `test_processed.csv` (dataset `train-processed`: the 1,000 unseen questions)
2. `question_neighbours.json` (dataset `nlp-qna-retrieval`: retrieval order)
3. `question_neighbours_reranked.json.gz` (Your Work -> re-ranker notebook 07 output)
4. the Phase 4 checkpoints (dataset `nlp-qna-adapters`: `nn/`, `none/`)
5. the final checkpoint (Your Work -> retrain notebook 08 output: `adapters/nn/final`)
6. *optional, saves about 5 GPU hours:* Your Work -> notebook 09 output. Its answers (300 test questions per system + the Xt answers)
   are reused **only if** the checkpoint file hash, the decoding settings and the exact prompt are identical; anything else is regenerated.
7. *only when resuming:* this notebook's own previous output (see "If it stops early" at the end).

| system | model | reference answers in the prompt |
|---|---|---|
| B0a (original) | none - copy the nearest training answer | - |
| B0a (re-ranked) | none - copy the top re-ranked training answer | - |
| B1 | Qwen2.5-3B-Instruct zero-shot | none |
| B2 | QLoRA adapter v1 `none` (4,000 rows) | none |
| P1 | QLoRA adapter v1 `nn` (4,000 rows) | top-2, original order |
| **P2 (final)** | **QLoRA adapter v2 (all 12,825 rows)** | **top-2, re-ranked** |

Every generated system uses the same decoding (chosen in notebook 06): greedy, repetition penalty 1.1 on the answer's own tokens,
no repeated 12-token n-gram in the answer, at most 1,280 new tokens.
**Xt:** exact repeats of a training question -> the stored medoid answer (router); every other question -> P2.

**Outputs:** `xt_predictions.csv` (submission), `test_predictions.csv` (final system, all 1,500 test questions),
`test_predictions_all_systems.csv`, `metrics.json`, `per_item_scores.csv`, `gen/` (every raw answer), `jobs.jsonl` (every prompt).
"""),
        code("!pip install -q -U peft bitsandbytes accelerate huggingface_hub rouge-score sacrebleu\n!pip install -q bert-score sentence-transformers"),
        code("%%writefile prompts.py\n" + src("prompts.py")),
        code("%%writefile eval_harness.py\n" + src("eval_harness.py")),
        code("%%writefile infer.py\n" + src("infer.py")),
        md("## 0. Configuration and checkpoints"),
        code('''
import glob, os, sys, json, gzip, csv, random, time, shutil, subprocess, hashlib, collections as C
csv.field_size_limit(10 ** 9)
_missing = [f for f in ("prompts.py", "eval_harness.py", "infer.py") if not os.path.exists(f)]
assert not _missing, "missing " + str(_missing) + " - use Run All from the top"
sys.path.insert(0, os.getcwd())
import torch
CPU = os.environ.get("P11_CPU") == "1"     # local smoke-test hook, never set on Kaggle
if not CPU:
    assert torch.cuda.is_available(), "No GPU. Session options -> Accelerator -> GPU T4 x2, then Run All."
N_GPU = torch.cuda.device_count()
print("GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])

MODEL = os.environ.get("P11_MODEL", "Qwen/Qwen2.5-3B-Instruct")
N_TEST = int(os.environ.get("P11_NTEST", "0"))          # 0 = all 1,500 test questions (smoke tests use fewer)
XT_LIMIT = int(os.environ.get("P11_XT_LIMIT", "0"))     # smoke tests only
DECODE = {"max_new": int(os.environ.get("P11_MAXNEW", "1280")), "rep_penalty": 1.0, "gen_rep_penalty": 1.1, "gen_no_repeat": 12}
BATCH = 10
GEN_HOURS = float(os.environ.get("P11_HOURS", "10.25"))  # workers stop cleanly after this (12 h Kaggle limit incl. scoring)
SPEED = {"base": 6.4, "none_v1": 16.4, "nn_v1": 14.4, "nn_v2": 14.7}   # s/answer measured in notebook 09 (load balancing only)
WORK = os.environ.get("P11_WORK", "/kaggle/working")
INPUT = os.environ.get("P11_INPUT", "/kaggle/input")

def find(pattern, required=True):
    hit = sorted(p.replace("\\\\", "/") for p in glob.glob(INPUT + "/**/" + pattern, recursive=True))
    if not hit and required:
        raise AssertionError("attach a dataset / notebook output containing " + pattern)
    return hit[0] if hit else None

def op(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

TRAIN_CSV, UNSEEN_CSV = find("train_processed*.csv"), find("test_processed*.csv")
NN_ORIG, NN_RR = find("question_neighbours.json*"), find("question_neighbours_reranked.json*")
V1_NN = os.path.dirname(find("nn/adapter_config.json"))
V1_NONE = os.path.dirname(find("none/adapter_config.json"))
V2_NN = os.path.dirname(find("adapters/nn/final/adapter_config.json"))
V2_SUMMARY = os.path.join(os.path.dirname(V2_NN), "train_summary.json")
V2_STATUS = json.load(open(V2_SUMMARY)) if os.path.exists(V2_SUMMARY) else {}
print("final checkpoint:", V2_NN, "| training status:", V2_STATUS.get("status"), "| steps:", V2_STATUS.get("steps"))

AD = WORK + "/ad"                                  # the three checkpoints under distinct names
for name, srcdir in (("nn_v1", V1_NN), ("none_v1", V1_NONE), ("nn_v2", V2_NN)):
    if not os.path.exists(AD + "/" + name):
        shutil.copytree(srcdir, AD + "/" + name)
AD_HASH = {n: md5(os.path.join(AD, n, "adapter_model.safetensors")) for n in ("nn_v1", "none_v1", "nn_v2")}
json.dump({"model": MODEL, "decoding": DECODE, "adapter_md5": AD_HASH}, open(WORK + "/run_config.json", "w"), indent=1)
print("checkpoint md5:", AD_HASH)
print("decoding:", DECODE)
'''),
        md("## 1. Router, test questions and prompts"),
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
print("unseen: %d | routed by exact match: %d | to generate: %d" % (len(unseen), len(routed), len(gen_unseen)))

all_test = [r for r in rows if r["split"] == "test"]
test_rows = random.Random(42).sample(all_test, N_TEST) if N_TEST else all_test
print("test questions (never seen in training): %d of %d" % (len(test_rows), len(all_test)))

# (name, adapter, prompt mode, neighbour lists)
SYSTEMS = [("B1 zero-shot base model", "base", "none", None),
           ("B2 fine-tuned v1, no retrieval", "none_v1", "none", None),
           ("P1 fine-tuned v1 + retrieval (Phase 4)", "nn_v1", "nn", NNO),
           ("P2 FINAL fine-tuned v2 + re-ranked retrieval", "nn_v2", "nn", NNR)]
FINAL = "P2 FINAL fine-tuned v2 + re-ranked retrieval"

def mk(adapter, prio, tag, qid, question, mode, NNs, split):
    ex = P.make_example(ctx, mode, qid, question, NNs[split].get(qid, []) if mode != "none" else [], tok=tok)
    return {"uid": "%s|%s|%s" % (adapter, tag, qid), "adapter": adapter, "prio": prio, "system": ex["system"], "user": ex["user"]}

jobs = [mk("nn_v2", 0, "xt", u["question_id"], u["question"], "nn", NNR, "xt") for u in gen_unseen]   # Xt first
for r in test_rows:
    for name, ad, mode, NNs in SYSTEMS:
        jobs.append(mk(ad, 1, "hold", r["question_id"], r["question"], mode, NNs, "test"))
os.makedirs(WORK + "/gen", exist_ok=True)
with open(WORK + "/jobs.jsonl", "w", encoding="utf-8") as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + "\\n")
print("jobs:", len(jobs), dict(C.Counter(j["adapter"] for j in jobs)))
'''),
        md("""
## 2. Reuse verified earlier answers (notebook 09 output / this notebook's previous run)
An earlier answer is reused only if **all** of these hold: same base model and decoding settings, the checkpoint file has the same
MD5 hash, and the prompt is character-for-character identical. Everything else is generated below.
"""),
        code('''
want = {j["uid"]: j for j in jobs}
reused, reuse_log = {}, []
for jf in sorted(glob.glob(INPUT + "/**/jobs.jsonl", recursive=True)):
    folder = os.path.dirname(jf)
    cfg = {}
    for f in ("run_config.json", "metrics.json"):
        if os.path.exists(os.path.join(folder, f)):
            cfg = json.load(open(os.path.join(folder, f)))
            break
    prev_model = cfg.get("model") or cfg.get("config", {}).get("MODEL")
    if cfg.get("decoding") != DECODE or prev_model != MODEL:
        reuse_log.append({"folder": folder, "used": 0, "reason": "different decoding or base model"}); continue
    ok = {n for n in AD_HASH if os.path.exists(os.path.join(folder, "ad", n, "adapter_model.safetensors"))
          and md5(os.path.join(folder, "ad", n, "adapter_model.safetensors")) == AD_HASH[n]}
    ok.add("base")                                     # base model = no adapter; same MODEL checked above
    prev_jobs = {}
    for l in open(jf, encoding="utf-8"):
        if l.strip():
            j = json.loads(l); prev_jobs[j["uid"]] = j
    used = rejected = 0
    for g in sorted(glob.glob(os.path.join(folder, "gen", "*.jsonl"))):
        for l in open(g, encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l); u = d["uid"]
            if u not in want or u in reused:
                continue
            pj = prev_jobs.get(u)
            if (d["adapter"] in ok and pj and pj["system"] == want[u]["system"] and pj["user"] == want[u]["user"]
                    and d["text"].strip()):
                reused[u] = d; used += 1
            else:
                rejected += 1
    reuse_log.append({"folder": folder, "used": used, "rejected": rejected, "verified_checkpoints": sorted(ok)})
with open(WORK + "/gen/reused.jsonl", "w", encoding="utf-8") as f:
    for d in reused.values():
        f.write(json.dumps(d, ensure_ascii=False) + "\\n")
for r in reuse_log:
    print("reuse:", r)
print("reused answers: %d of %d jobs %s" % (len(reused), len(jobs), dict(C.Counter(d["adapter"] for d in reused.values()))))
'''),
        md("## 3. Generate - jobs balanced over the GPUs (resumable)"),
        code('''
if not CPU:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
done_before = set()
for f in glob.glob(WORK + "/gen/gen_gpu*.jsonl"):                 # same-session re-run: keep what is there
    done_before |= {json.loads(l)["uid"] for l in open(f, encoding="utf-8") if l.strip()}
todo = [j for j in jobs if j["uid"] not in reused and j["uid"] not in done_before]
n_workers = 2 if N_GPU >= 2 else 1
loads, assign = [0.0] * n_workers, [[] for _ in range(n_workers)]
for j in sorted(todo, key=lambda j: (j["prio"], -SPEED[j["adapter"]])):     # greedy: each job to the least-loaded GPU
    k = loads.index(min(loads)); assign[k].append(j); loads[k] += SPEED[j["adapter"]]
print("to generate: %d | estimated hours per GPU: %s" % (len(todo), [round(x / 3600, 1) for x in loads]))
procs = []
for k in range(n_workers):
    if not assign[k]:
        continue
    jf = "%s/jobs_gpu%d.jsonl" % (WORK, k)
    with open(jf, "w", encoding="utf-8") as f:
        for j in assign[k]:
            f.write(json.dumps(j, ensure_ascii=False) + "\\n")
    names = [n for n in ("nn_v2", "nn_v1", "none_v1", "base") if any(j["adapter"] == n for j in assign[k])]   # nn_v2 first: Xt first
    log = open("%s/gen/worker%d.log" % (WORK, k), "a")
    cmd = [sys.executable, "infer.py", "--jobs", jf, "--adapter_dir", AD, "--adapters", ",".join(names),
           "--out", "%s/gen/gen_gpu%d.jsonl" % (WORK, k), "--model", MODEL, "--batch", str(BATCH), "--hours", str(GEN_HOURS),
           "--max_new", str(DECODE["max_new"]), "--rep_penalty", str(DECODE["rep_penalty"]),
           "--gen_rep_penalty", str(DECODE["gen_rep_penalty"]), "--gen_no_repeat", str(DECODE["gen_no_repeat"])]
    if CPU:
        cmd.append("--no_quant")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
    if not CPU:
        env["CUDA_VISIBLE_DEVICES"] = str(k)
    procs.append((k, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    print("started worker", k, names, len(assign[k]), "jobs")
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(120 if not CPU else 2)
    for k, p in procs:
        lines = open("%s/gen/worker%d.log" % (WORK, k), errors="ignore").read().splitlines()
        print("[%4.0f min] worker %d: %s" % ((time.time() - t0) / 60, k, lines[-1][:150] if lines else "(starting)"), flush=True)
for k, p in procs:
    print("worker", k, "exit code", p.returncode)
    if p.returncode:
        print(open("%s/gen/worker%d.log" % (WORK, k), errors="ignore").read()[-3000:])
'''),
        md("## 4. Collect the answers"),
        code('''
import re
gen = {}
for f in glob.glob(WORK + "/gen/*.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); gen[d["uid"]] = d
missing = [j["uid"] for j in jobs if j["uid"] not in gen]
xt_missing = [u for u in missing if "|xt|" in u]
COMPLETE = not missing
print("answers: %d of %d jobs (%d reused, %d generated) | missing: %d (Xt: %d)" % (
    len(jobs) - len(missing), len(jobs), len(reused), len(jobs) - len(missing) - len(reused), len(missing), len(xt_missing)))
print("STATUS:", "COMPLETE" if COMPLETE else "INCOMPLETE - see 'If it stops early' at the end")
WORD = re.compile(r"\\S+")
def is_loop(text, n=8, min_rep=3):
    w = [x.lower().strip(".,;:*#-()") for x in WORD.findall(text)]
    c = C.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return bool(c) and max(c.values()) >= min_rep
'''),
        md("## 5. Test-set evaluation: lexical metrics"),
        code('''
import numpy as np, pandas as pd
import eval_harness as H
def nearest(NNs, split, q):
    nb = NNs[split].get(q)
    return qid2row[nb[0][0]]["final_answer"] if nb else ""

ids = [r["question_id"] for r in test_rows if all(("%s|hold|%s" % (ad, r["question_id"])) in gen for _, ad, _, _ in SYSTEMS)]
refs = {q: qid2row[q]["final_answer"] for q in ids}
B0O, B0R = "B0a copy nearest answer (original order)", "B0a copy nearest answer (re-ranked)"
systems = {B0O: {q: nearest(NNO, "test", q) for q in ids}, B0R: {q: nearest(NNR, "test", q) for q in ids}}
extra = {B0O: {}, B0R: {}}
for name, ad, mode, NNs in SYSTEMS:
    p, e, cap = {}, 0, 0
    for q in ids:
        d = gen["%s|hold|%s" % (ad, q)]
        cap += d["hit_limit"]
        if d["text"].strip():
            p[q] = d["text"]
        else:
            p[q], e = nearest(NNR, "test", q), e + 1     # empty generation -> nearest answer (counted)
    systems[name] = p
    extra[name] = {"empty_replaced": e, "hit_cap": cap}
SYS = list(systems)
res, per, boot = {n: {} for n in SYS}, {}, {}
for n, p in (systems.items() if ids else []):
    res[n] = H.evaluate(p, refs)
    res[n].update(extra[n])
    res[n]["loops_8gram_x3"] = sum(is_loop(p[q]) for q in ids)
    res[n]["median_words"] = float(np.median([len(p[q].split()) for q in ids]))
print("test questions scored: %d of %d%s" % (len(ids), len(test_rows), "" if len(ids) == len(test_rows) else "  <-- PARTIAL"))
if not ids:
    print("no test question has answers from every system yet - scoring is skipped in this session")
'''),
        md("## 6. Test-set evaluation: semantic metrics (BERTScore, bge-m3)"),
        code('''
if ids:
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer
    from rouge_score import rouge_scorer
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    R = [refs[q] for q in ids]
    per = {"rougeL": {}, "bertscore_f1": {}, "bertscore_f1_rescaled": {}, "cos_answer_reference": {}}
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for s in SYS:
        per["rougeL"][s] = np.array([rs.score(refs[q], systems[s][q])["rougeL"].fmeasure for q in ids])
    t0 = time.time()
    scorer = BERTScorer(model_type="roberta-large", num_layers=17, lang="en", rescale_with_baseline=True, device=DEV, batch_size=16)
    base_f = float(scorer.baseline_vals[2])
    for s in SYS:
        _, _, F = scorer.score([systems[s][q] for q in ids], R)
        F = F.cpu().numpy().astype(float)
        per["bertscore_f1_rescaled"][s], per["bertscore_f1"][s] = F, F * (1 - base_f) + base_f
    del scorer
    if DEV == "cuda":
        torch.cuda.empty_cache()
    emb = SentenceTransformer("BAAI/bge-m3", device=DEV)
    emb.max_seq_length = 2048
    if DEV == "cuda":
        emb.half()
    enc = lambda texts: emb.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    E_ref = enc(R)
    for s in SYS:
        per["cos_answer_reference"][s] = (enc([systems[s][q] for q in ids]) * E_ref).sum(1).astype(float)
    del emb
    print("semantic scoring done in %.1f min" % ((time.time() - t0) / 60))
    for s in SYS:
        for m in ("bertscore_f1", "bertscore_f1_rescaled", "cos_answer_reference"):
            res[s][m] = float(per[m][s].mean())
'''),
        md("## 7. Results table and paired bootstraps"),
        code('''
if ids:
    cols = ["rougeL", "rouge1", "rouge2", "bleu", "fact_f1", "bertscore_f1", "bertscore_f1_rescaled", "cos_answer_reference",
            "len_ratio_mean", "median_words", "hit_cap", "loops_8gram_x3", "empty_replaced"]
    table = pd.DataFrame({n: {c: res[n].get(c, np.nan) for c in cols} for n in SYS}).T
    pd.set_option("display.width", 300)
    print("N = %d test questions (reference median %d words)" % (len(ids), np.median([len(refs[q].split()) for q in ids])))
    print(table.round(4).to_string())

    n = len(ids)
    idx = np.random.default_rng(0).integers(0, n, (2000, n))
    def diff_ci(m, a, b):
        d = per[m][a] - per[m][b]
        bd = d[idx].mean(1)
        return {"diff": float(d.mean()), "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
                "a_wins": int((d > 0).sum()), "b_wins": int((d < 0).sum())}
    B1N, B2N, P1N = "B1 zero-shot base model", "B2 fine-tuned v1, no retrieval", "P1 fine-tuned v1 + retrieval (Phase 4)"
    PAIRS = [(FINAL, B0R, "final - copy re-ranked nearest answer"), (FINAL, B0O, "final - copy nearest answer (original order)"),
             (FINAL, P1N, "final - Phase 4 model"), (FINAL, B2N, "final - no retrieval"), (FINAL, B1N, "final - zero-shot"),
             (B0R, B0O, "re-ranked copy - original copy"), (P1N, B0O, "Phase 4 model - original copy")]
    boot = {m: {lab: diff_ci(m, a, b) for a, b, lab in PAIRS} for m in ("rougeL", "bertscore_f1", "cos_answer_reference")}
    for m, d in boot.items():
        print("\\n%s  (paired bootstrap, n=%d, 2000 resamples)" % (m, n))
        for lab, r in d.items():
            flag = "  *" if r["lo"] > 0 or r["hi"] < 0 else ""
            print("  %-46s %+.4f  [95%% CI %+.4f, %+.4f]  wins %d vs %d%s" % (lab, r["diff"], r["lo"], r["hi"], r["a_wins"], r["b_wins"], flag))
    print("\\n* = the confidence interval excludes 0")

    # consistency check with notebook 09 (same seed-42 sample of 300)
    h300 = [r["question_id"] for r in random.Random(42).sample(all_test, 300)] if len(all_test) >= 300 else []
    m09 = [f for f in glob.glob(INPUT + "/**/metrics.json", recursive=True) if json.load(open(f)).get("holdout_n") == 300]
    if h300 and set(h300) <= set(ids) and m09:
        ref09 = json.load(open(m09[0]))["systems"]
        print("\\nsame 300 questions as notebook 09 - ROUGE-L here vs notebook 09:")
        pos = {q: i for i, q in enumerate(ids)}
        for s in SYS:
            if s in ref09:
                v = per["rougeL"][s][[pos[q] for q in h300]].mean()
                print("  %-46s %.4f vs %.4f %s" % (s, v, ref09[s]["rougeL"], "(identical)" if abs(v - ref09[s]["rougeL"]) < 1e-9 else ""))
'''),
        md("## 8. The router, scored separately\nOnly repeated questions can be routed, so this is measured on training questions that appear more than once: the stored answer is compared with a *different* human answer to the same question."),
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
print("repeated-question groups: %d | router (medoid) rougeL %.4f | random other human answer %.4f (conservative)" % (
    len(held), router_res["rougeL"], rand_res["rougeL"]))
'''),
        md("## 9. Write the prediction files"),
        code('''
sfx = "" if COMPLETE else "_PARTIAL"
def write(path, header, items, rowf):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for it in items:
            w.writerow(rowf(it))

# unseen set (Xt): router, else the final model
final, source, baseline = {}, {}, {}
for u in unseen:
    q = u["question_id"]
    if q in routed:
        final[q] = baseline[q] = routed[q]; source[q] = "router"
        continue
    baseline[q] = nearest(NNR, "xt", q)
    d = gen.get("nn_v2|xt|%s" % q)
    if d and d["text"].strip():
        final[q], source[q] = d["text"], "generated"
    else:
        final[q], source[q] = baseline[q], "fallback_nearest"
xt_ok = not xt_missing and len(final) == len(unseen) == 1000 and all(v.strip() for v in final.values())
xs = "" if xt_ok else "_PARTIAL"
write(WORK + "/xt_predictions%s.csv" % xs, ["question_id", "question", "final_answer"], unseen,
      lambda u: [u["question_id"], u["question"], final[u["question_id"]]])
write(WORK + "/xt_predictions_detailed%s.csv" % xs, ["question_id", "source", "final_answer"], unseen,
      lambda u: [u["question_id"], source[u["question_id"]], final[u["question_id"]]])
write(WORK + "/xt_predictions_with_baseline%s.csv" % xs, ["question_id", "question", "final_answer", "baseline_answer"], unseen,
      lambda u: [u["question_id"], u["question"], final[u["question_id"]], baseline[u["question_id"]]])
print("Xt:", "complete" if xt_ok else "INCOMPLETE", "|", dict(C.Counter(source.values())))

# test set: the final system for every test question, and all systems side by side
scored = [qid2row[q] for q in ids]
write(WORK + "/test_predictions%s.csv" % sfx, ["question_id", "question", "reference_answer", "final_answer"], scored,
      lambda r: [r["question_id"], r["question"], r["final_answer"], systems[FINAL][r["question_id"]]])
write(WORK + "/test_predictions_all_systems%s.csv" % sfx, ["question_id", "question", "reference_answer"] + SYS, scored,
      lambda r: [r["question_id"], r["question"], r["final_answer"]] + [systems[s][r["question_id"]] for s in SYS])
flat = pd.DataFrame({"question_id": ids})
for m in per:
    for s in SYS:
        flat["%s|%s" % (m, s)] = per[m][s]
flat.to_csv(WORK + "/per_item_scores%s.csv" % sfx, index=False)
xt_gen = [gen[u] for u in gen if u.startswith("nn_v2|xt|")]
json.dump({"status": "complete" if COMPLETE else "incomplete", "test_n": len(test_rows), "test_scored": len(ids),
           "decoding": DECODE, "model": MODEL, "adapter_md5": AD_HASH, "reuse": reuse_log, "n_reused": len(reused),
           "systems": res, "bootstrap": boot, "router": router_res, "router_random_other": rand_res,
           "routed_share": len(routed) / len(unseen),
           "unseen": {"sources": dict(C.Counter(source.values())), "generated_hit_cap": sum(d["hit_limit"] for d in xt_gen),
                      "generated_loops_8gram_x3": sum(is_loop(d["text"]) for d in xt_gen)},
           "final_checkpoint_training": {k: v for k, v in V2_STATUS.items() if k not in ("loss_log", "eval_log")}},
          open(WORK + "/metrics%s.json" % sfx, "w"), indent=1, default=float)
print("wrote test_predictions%s.csv (%d rows), test_predictions_all_systems%s.csv, per_item_scores%s.csv, metrics%s.json" % (sfx, len(ids), sfx, sfx, sfx))
print("\\nSTATUS:", "COMPLETE - all files final" if COMPLETE else "INCOMPLETE - files are marked _PARTIAL; see below")
'''),
        md("## 10. A few answers side by side"),
        code('''
for q in ids[:3]:
    print("=" * 100, "\\nQ:", qid2row[q]["question"][:200])
    print("--- reference :", refs[q][:400].replace("\\n", " "))
    print("--- final (P2):", systems[FINAL][q][:400].replace("\\n", " "))
    print("--- B0a re-ranked copy:", systems[B0R][q][:250].replace("\\n", " "))
'''),
        md("""
## If it stops early
Generation stops cleanly after 10.25 hours so the scoring still fits in Kaggle's 12-hour limit. If the status says **INCOMPLETE**
(files end in `_PARTIAL`): open a new version of this notebook, **Add Input -> Your Work -> this notebook's output** (keep all the
other inputs), and **Save & Run All** again. The previous answers are verified and reused, and only the missing ones are generated.

## After it finishes
Send me the notebook link (or the downloaded output). Output tab: `xt_predictions.csv` (submission), `test_predictions.csv`,
`test_predictions_all_systems.csv`, `metrics.json`, `per_item_scores.csv`, `gen/`, `jobs.jsonl`.
"""),
    ]
    return notebook(cells)


if __name__ == "__main__":
    p = os.path.join(ROOT, "notebooks", "11_final_inference_full_test.ipynb")
    json.dump(build(), open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p, round(os.path.getsize(p) / 1e6, 2), "MB")
