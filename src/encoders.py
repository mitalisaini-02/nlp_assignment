"""Thin wrapper around sentence-transformers so the notebook (and tests) can swap the encoder easily.
Pure ASCII source on purpose."""
import numpy as np

_MODELS = {}


def encode(model_name, texts, batch_size=64, max_len=None, prefix="", fp16=True):
    """L2-normalised embeddings as float16 (cosine similarity == dot product)."""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name not in _MODELS:
        _MODELS.clear()
        torch.cuda.empty_cache()
        m = SentenceTransformer(model_name, device=dev)
        if fp16 and dev == "cuda":
            m.half()
        _MODELS[model_name] = m
    m = _MODELS[model_name]
    if max_len:
        m.max_seq_length = max_len
    if prefix:
        texts = [prefix + t for t in texts]
    E = m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=True)
    return E.astype(np.float16)
