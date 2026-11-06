"""Build the document manifest consumed by the Kaggle download notebook.

Output: data/processed/sources_manifest.json
  {"docs": [{doc_id, kind, url, names, source_types, num_questions, num_citations, pages_cited}, ...],
   "skipped": {...counts...}}
Docs are sorted by num_questions (desc) so the most-cited PDFs are fetched first.
"""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from sources import parse_sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "training_set_15000.csv"), encoding="utf-8")))
    docs = {}
    skipped = defaultdict(int)
    for r in rows:
        qid = r["question_id"]
        for c in parse_sources(r["final_answer_sources"]):
            if c["kind"] == "zoho":
                key = c["file_id"]
            elif c["kind"] == "external":
                key = "ext_" + hashlib.md5(c["url"].encode()).hexdigest()[:16]
            else:
                skipped[c["kind"]] += 1
                continue
            d = docs.setdefault(key, {
                "doc_id": key, "kind": c["kind"], "url": c["url"],
                "names": set(), "source_types": set(), "qids": set(),
                "num_citations": 0, "pages_cited": set(),
            })
            if c["name"]:
                d["names"].add(c["name"])
            if c["source_type"]:
                d["source_types"].add(c["source_type"])
            d["qids"].add(qid)
            d["num_citations"] += 1
            d["pages_cited"].update(c["pages"])
    out = []
    for d in docs.values():
        out.append({
            "doc_id": d["doc_id"], "kind": d["kind"], "url": d["url"],
            "names": sorted(d["names"]), "source_types": sorted(d["source_types"]),
            "num_questions": len(d["qids"]), "num_citations": d["num_citations"],
            "pages_cited": sorted(d["pages_cited"]),
        })
    out.sort(key=lambda d: (-d["num_questions"], d["doc_id"]))
    os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
    path = os.path.join(ROOT, "data", "processed", "sources_manifest.json")
    json.dump({"docs": out, "skipped": dict(skipped)}, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    kinds = defaultdict(int)
    for d in out:
        kinds[d["kind"]] += 1
    print("docs", len(out), dict(kinds), "skipped", dict(skipped))
    print("cumulative question coverage by top-N docs (share of rows citing >=1 top-N doc):")
    qsets = {}
    for r in rows:
        qsets[r["question_id"]] = {k for k in (
            (c["file_id"] if c["kind"] == "zoho" else None) for c in parse_sources(r["final_answer_sources"])) if k}
    for n in (100, 300, 500, 1000, 3000):
        top = {d["doc_id"] for d in out[:n]}
        print(" top", n, round(sum(1 for s in qsets.values() if s & top) / len(rows), 3))
    print("size KB", os.path.getsize(path) // 1024)


if __name__ == "__main__":
    main()
