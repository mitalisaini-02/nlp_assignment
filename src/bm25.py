"""Sparse BM25 (Okapi) that scales to hundreds of thousands of documents.

Term weights are precomputed into a sparse matrix, so scoring a batch of queries is one sparse matrix product.
Pure ASCII source on purpose.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1=1.2, b=0.75, stop_words="english"):
        self.k1, self.b, self.stop_words = k1, b, stop_words

    def fit(self, texts):
        self.cv = CountVectorizer(lowercase=True, stop_words=self.stop_words, token_pattern=r"(?u)\b\w+\b",
                                  dtype=np.float32)
        X = self.cv.fit_transform(texts).tocsr()
        n = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        avg = max(1e-9, float(dl.mean()))
        rows = np.repeat(np.arange(n), np.diff(X.indptr))
        tf = X.data
        X.data = (idf[X.indices] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl[rows] / avg))
                  ).astype(np.float32)
        self.W = X.T.tocsr()  # (vocab, n_docs)
        self.n = n
        return self

    def search(self, queries, k=50, batch=128):
        """Returns (idx, score), each (n_queries, k); idx is -1 where a query shares no term with the document."""
        Q = self.cv.transform(queries).tocsr()
        Q.data[:] = 1.0
        k = min(k, self.n)
        idx = np.full((Q.shape[0], k), -1, dtype=np.int32)
        sc = np.zeros((Q.shape[0], k), dtype=np.float32)
        for i in range(0, Q.shape[0], batch):
            S = (Q[i:i + batch] @ self.W).toarray()
            part = np.argpartition(-S, k - 1, axis=1)[:, :k]
            ps = np.take_along_axis(S, part, axis=1)
            order = np.argsort(-ps, axis=1)
            idx[i:i + batch] = np.take_along_axis(part, order, axis=1)
            sc[i:i + batch] = np.take_along_axis(ps, order, axis=1)
        idx[sc <= 0] = -1
        return idx, sc
