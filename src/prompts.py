"""Phase 4: build the training / inference prompts.

Modes
  none : question only                                   (baseline B2: plain fine-tuning)
  nn   : question + the 2 nearest training Q&A pairs      (proposed model without PDFs)
  full : nn + source-page excerpts                         (proposed model with PDFs)

Neighbours for TRAIN questions already exclude the question's own group (see Phase 2), so the model is trained to
synthesise an answer from related-but-different answers, not to copy an identical one.
Pure ASCII source on purpose.
"""
import json
import random

SYSTEM = (
    "You are an expert Indian agricultural advisor. Answer the farmer's question in English with a detailed, "
    "well-structured advisory: key facts or symptoms, thresholds where relevant, cultural, biological and chemical "
    "management with exact doses and units, and safety notes. Use the reference material when it applies to this "
    "question; never invent doses."
)


def trim_words(text, n):
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def trim_tokens(text, n, tok):
    """Cut text to n tokens. Token budgets (not word counts) keep every prompt inside the length limit: numeric-heavy
    text can use twice as many tokens per word as prose."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tok.decode(ids[:n]) + " ..."


def build_user(mode, question, refs, excerpts, ref_words=300, pdf_words=180, tok=None, ref_tokens=380, pdf_tokens=200):
    """refs: list of (question, answer); excerpts: list of (label, text). With tok given, trimming is by tokens."""
    def cut(text, words, tokens):
        return trim_tokens(text, tokens, tok) if tok is not None else trim_words(text, words)
    if mode == "none":
        return "Question: " + question
    parts = ["Reference answers to similar questions:"]
    for i, (q, a) in enumerate(refs, 1):
        parts.append("[%d] Q: %s\nA: %s" % (i, q, cut(a, ref_words, ref_tokens)))
    if mode == "full" and excerpts:
        parts.append("Source excerpts:")
        for label, t in excerpts:
            parts.append("[%s] %s" % (label, cut(t, pdf_words, pdf_tokens)))
    parts.append("Question: " + question)
    return "\n\n".join(parts)


class Context:
    """Holds the lookups needed to build prompts for any query set."""

    def __init__(self, rows, page_text=None, chunk_text=None):
        self.qid2row = {r["question_id"]: r for r in rows}
        self.rows = rows
        self.page_text = page_text or {}      # (doc_id, page) -> first chunk text of that page
        self.chunk_text = chunk_text or []    # chunk id -> text
        self._cit = {}

    def citations(self, qid):
        if qid not in self._cit:
            self._cit[qid] = json.loads(self.qid2row[qid]["citations_json"])
        return self._cit[qid]

    def propagated_pages(self, neighbour_ids, limit=2):
        """Pages cited by the nearest training questions, in neighbour order. Returns [(label, text)]."""
        out, seen = [], set()
        for nid in neighbour_ids:
            for c in self.citations(nid):
                if c.get("kind") != "zoho":
                    continue
                for p in c.get("pages", []):
                    key = (c["doc_id"], p)
                    if key in self.page_text and key not in seen:
                        seen.add(key)
                        name = (c.get("name") or "source")[:60]
                        out.append(("%s, p.%d" % (name, p), self.page_text[key]))
                        if len(out) == limit:
                            return out
        return out


def make_example(ctx, mode, qid, question, neighbours, chunk_hits=None, answer=None, noise=0.0, rng=None,
                 pool_qids=None, n_refs=2, tok=None):
    """neighbours: [[question_id, score], ...] best first."""
    rng = rng or random.Random(0)
    nids = [n[0] for n in neighbours]
    refs_ids = nids[:n_refs]
    if noise and pool_qids and len(refs_ids) == n_refs and rng.random() < noise:
        refs_ids[-1] = rng.choice(pool_qids)      # RAFT-style distractor: the model learns to ignore bad context
    refs = [(ctx.qid2row[i]["question"], ctx.qid2row[i]["final_answer"]) for i in refs_ids]
    excerpts = []
    if mode == "full":
        excerpts = ctx.propagated_pages(nids[:3])
        if len(excerpts) < 2 and chunk_hits and ctx.chunk_text:
            for cid in chunk_hits:
                if len(excerpts) == 2:
                    break
                excerpts.append(("retrieved passage", ctx.chunk_text[cid]))
        if noise and excerpts and pool_qids and rng.random() < noise:
            other = ctx.propagated_pages([rng.choice(pool_qids)], limit=1)
            if other:
                excerpts[-1] = other[0]
    ex = {"qid": qid, "mode": mode, "system": SYSTEM, "user": build_user(mode, question, refs, excerpts, tok=tok)}
    if answer is not None:
        ex["answer"] = answer
    return ex
