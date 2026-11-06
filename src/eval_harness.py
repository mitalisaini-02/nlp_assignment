"""Single evaluation harness used by every system (baselines, ablations, final model).

    from eval_harness import evaluate
    scores = evaluate(preds, refs)            # dicts {question_id: text}
    scores = evaluate(preds, refs, bertscore=True, meteor=True)   # slower extras

Metrics: ROUGE-1/2/L F1, corpus BLEU (sacrebleu), length ratio, agronomic fact recall/precision/F1
(number+unit pairs, formulation-tagged chemicals such as "dinotefuran 20 SG", binomials), and optionally
METEOR (nltk) and BERTScore-F1. Extras are lazy-imported so the core runs with rouge-score + sacrebleu only.
"""
import re
import statistics as st

from rouge_score import rouge_scorer

_UNIT = {"gm": "g", "gram": "g", "grams": "g", "gms": "g", "litre": "l", "litres": "l", "lit": "l", "liter": "l",
         "liters": "l", "hectare": "ha", "hectares": "ha", "per cent": "%", "percent": "%", "quintals": "q",
         "quintal": "q", "tonnes": "t", "tonne": "t", "day": "days", "kgs": "kg"}
NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|\u2013|to)?\s*(\d+(?:\.\d+)?)?\s*"
    r"(kg|kgs|gms?|grams?|g|mg|ml|litres?|liters?|lit|l|quintals?|q|tonnes?|t|ha|hectares?|acre|cm|mm|"
    r"days?|per\s*cent|percent|%|ppm)\b"
    r"(?:\s*(?:/|per)\s*(ha|hectare|acre|litres?|liters?|lit|l|plant|kg|m2|10\s*l))?", re.I)
CHEM_RE = re.compile(r"\b([A-Za-z][A-Za-z\-]{3,})\s+\d+(?:\.\d+)?\s*%?\s*"
                     r"(SG|SL|EC|WP|WG|SC|FS|DP|GR|ZC|OD|EW|CS|ME|WS|DF)\b")
BINOM_RE = re.compile(r"\(([A-Z][a-z]{3,}\s+[a-z]{4,})\)")


def _u(x):
    x = re.sub(r"\s+", " ", x.lower().strip())
    return _UNIT.get(x, x)


def extract_facts(text):
    facts = set()
    for m in NUM_UNIT_RE.finditer(text):
        a, b, unit, per = m.groups()
        facts.add("%s%s%s%s" % (a, "-" + b if b else "", _u(unit), "/" + _u(per) if per else ""))
    for m in CHEM_RE.finditer(text):
        facts.add("chem:" + m.group(1).lower())
    for m in BINOM_RE.finditer(text):
        facts.add("sp:" + m.group(1).lower())
    return facts


def _fact_scores(preds, refs):
    r_list, p_list, f_list = [], [], []
    for p, r in zip(preds, refs):
        fr, fp = extract_facts(r), extract_facts(p)
        if not fr:
            continue
        inter = len(fr & fp)
        rec = inter / len(fr)
        prec = inter / len(fp) if fp else 0.0
        r_list.append(rec)
        p_list.append(prec)
        f_list.append(0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec))
    n = max(1, len(r_list))
    return {"fact_recall": sum(r_list) / n, "fact_precision": sum(p_list) / n, "fact_f1": sum(f_list) / n,
            "fact_n_refs": len(r_list)}


def evaluate(preds, refs, bertscore=False, meteor=False, bert_model="roberta-large"):
    ids = [i for i in refs if i in preds]
    P = [preds[i] or "" for i in ids]
    R = [refs[i] for i in ids]
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(P, R):
        s = sc.score(r, p)
        for k in agg:
            agg[k] += s[k].fmeasure
    n = max(1, len(ids))
    out = {"n": len(ids), "n_missing": len(refs) - len(ids)}
    out.update({k: v / n for k, v in agg.items()})
    import sacrebleu
    out["bleu"] = sacrebleu.corpus_bleu(P, [R]).score / 100.0
    pl, rl = [len(p.split()) for p in P], [len(r.split()) for r in R]
    out["len_ratio_mean"] = (sum(pl) / n) / max(1e-9, sum(rl) / n)
    out["len_ratio_median_per_item"] = st.median([a / max(1, b) for a, b in zip(pl, rl)]) if ids else 0.0
    out.update(_fact_scores(P, R))
    if meteor:
        from nltk.translate.meteor_score import meteor_score
        out["meteor"] = sum(meteor_score([r.split()], p.split()) for p, r in zip(P, R)) / n
    if bertscore:
        from bert_score import score as bs
        _, _, f1 = bs(P, R, model_type=bert_model, lang="en", batch_size=16, verbose=False)
        out["bertscore_f1"] = float(f1.mean())
    return out


def evaluate_by_group(preds, refs, groups, **kw):
    """groups: {name: set(question_ids)} -> {name: scores}; e.g. novel vs repeated, or per crop."""
    return {g: evaluate(preds, {i: refs[i] for i in ids if i in refs}, **kw) for g, ids in groups.items()}


if __name__ == "__main__":
    t = "Spray dinotefuran 20 SG @ 60 g/acre; seed rate 25 kg/ha. Pest: Amrasca devastans (Amrasca devastans)."
    print(extract_facts(t))
    print(evaluate({"a": t}, {"a": t}))
