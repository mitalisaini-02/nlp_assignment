"""Phase 1 worker: recompute page quality from the saved text, decide the document verdict, then chunk.

Lives in its own module so multiprocessing can import it (works with both fork and spawn).
Source is pure ASCII on purpose.
"""
import chunking
import pdf_pipeline as pp


def requalify_and_chunk(item):
    """item = (doc_id, pages). Returns (doc_id, verdict, n_pages, stored_verdict_counts, chunks, kept_pages, sample)."""
    doc_id, pages = item
    for p in pages:
        stats = pp.page_stats(p["text"])  # same function the downloader uses, now with the fixed weird-char test
        p.update(stats)
    verdict = pp.verdict(pages)
    chunks = chunking.process_doc(pages, verdict)
    qcounts = {}
    for p in pages:
        qcounts[p["quality"]] = qcounts.get(p["quality"], 0) + 1
    # one short example page per doc, used to eyeball excluded documents
    bad = [p for p in pages if p["quality"] == "garbled"]
    sample = None
    if verdict in ("garbled_legacy_font", "mixed") and bad:
        q = bad[len(bad) // 2]
        sample = {"page": q["page"], "sw": q["sw"], "text": q["text"][:300]}
    return doc_id, verdict, len(pages), qcounts, chunks, sample
