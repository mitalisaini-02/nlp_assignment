"""Code shared by 01_train.ipynb and 02_inference.ipynb. Change settings here so both notebooks stay in step."""
import ast
import gc
import hashlib
import io
import json
import re
import time
import tokenize
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

SEED = 42
DRIVE_ROOT = Path("/content/drive/MyDrive/agricultural_qa")
EMBED_MODEL = "intfloat/multilingual-e5-small"
GEN_MODEL = "google/flan-t5-small"
INSTRUCTION = (
    "Answer the farmer's question using the context. Preserve scientific names, chemical names, "
    "dosages, numbers and units. Do not invent recommendations."
)

# Changing any of these starts a new run folder, so old files can never be mixed in.
CHUNK_WORDS = 120
CHUNK_OVERLAP = 20
TOP_K = 3
MAX_SOURCE_TOKENS = 512
MAX_TARGET_TOKENS = 256
TRAIN_BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-4
LORA_R = 16
LORA_ALPHA = 32

# These can change at any time without retraining.
GEN_BATCH_SIZE = 32
MAX_NEW_TOKENS = 192
TRAIN_BUDGET_MINUTES = 55  # 01_train stops training at this much total notebook time


class Paths:
    """Where every file lives. The run ID comes from the settings above and the size of the CSV."""

    def __init__(self, smoke=False):
        self.train_csv = DRIVE_ROOT / "data" / "training_set_15000.csv"
        self.unseen_csv = DRIVE_ROOT / "data" / "unseen_set_1000.csv"
        for path in (self.train_csv, self.unseen_csv):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Upload it under exactly this name.")

        settings = [self.train_csv.stat().st_size, EMBED_MODEL, GEN_MODEL, INSTRUCTION, CHUNK_WORDS, CHUNK_OVERLAP, TOP_K,
                    MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, TRAIN_BATCH_SIZE, EPOCHS, LEARNING_RATE, LORA_R, LORA_ALPHA]
        run_id = hashlib.md5(json.dumps(settings).encode()).hexdigest()[:8]
        self.run = DRIVE_ROOT / ("fast_v1_smoke" if smoke else "fast_v1") / run_id
        self.data = self.run / "data"
        self.index = self.run / "index"
        self.prompts = self.run / "prompts"
        self.checkpoints = self.run / "checkpoints"
        self.model = self.run / "model"
        self.outputs = self.run / "outputs"
        for folder in (self.data, self.index, self.prompts, self.checkpoints, self.model, self.outputs):
            folder.mkdir(parents=True, exist_ok=True)
        print("Run folder:", self.run)


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Choose Runtime > Change runtime type > T4 GPU, then run again.")
    print("GPU:", torch.cuda.get_device_name(0))


def free_gpu():
    """Call after `del` on the model, otherwise nothing is released."""
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def stage(name, paths):
    """Print how long a stage took and save it to outputs/timings.json."""
    start = time.time()
    print(f"\n--- {name} ---")
    yield
    seconds = round(time.time() - start, 1)
    print(f"[{name}] {seconds} s | GPU memory in use: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    timings_path = paths.outputs / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    timings[name] = seconds
    timings_path.write_text(json.dumps(timings, indent=2))


# ---------------------------------------- Text helpers ----------------------------------------
def parse_source_names(raw):
    """Read source names from the numpy-style `final_answer_sources` string."""
    try:
        items = ast.literal_eval(raw)
        if not (isinstance(items, list) and all(isinstance(x, dict) for x in items)):
            items = None
    except (ValueError, SyntaxError, TypeError):
        items = None

    if items is None:
        try:
            strings = [t.string for t in tokenize.generate_tokens(io.StringIO(str(raw)).readline) if t.type == tokenize.STRING]
        except (tokenize.TokenError, SyntaxError):
            strings = []
        items = []
        for s in strings:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, str):
                    parsed = ast.literal_eval(parsed)
                if isinstance(parsed, dict):
                    items.append(parsed)
            except (ValueError, SyntaxError, TypeError):
                pass
    return [d["sourceName"] for d in items if d.get("sourceName")]


def detect_language(text):
    """Gurmukhi script means Punjabi, Devanagari means Hindi, anything else counts as English."""
    if re.search(r"[਀-੿]", str(text)):
        return "pa"
    if re.search(r"[ऀ-ॿ]", str(text)):
        return "hi"
    return "en"


def words(text):
    return re.findall(r"[\wऀ-ॿ਀-੿]+", str(text).lower())


def make_prompt(question, context):
    """The context comes last, so cutting a long prompt at 512 tokens can only shorten the context."""
    return f"{INSTRUCTION}\nquestion: {question}\ncontext: {context}"


# ---------------------------------------- Data ----------------------------------------
def load_data(paths, smoke=False):
    """Process the CSV files once. After that, load the saved parquet files."""
    files = {name: paths.data / f"{name}.parquet" for name in ("train", "val", "test", "unseen")}
    if all(f.exists() for f in files.values()):
        return {name: pd.read_parquet(f) for name, f in files.items()}

    df = pd.read_csv(paths.train_csv, dtype=str, keep_default_na=False)
    unseen = pd.read_csv(paths.unseen_csv, dtype=str, keep_default_na=False)
    unseen["in_training_file"] = unseen["question"].isin(set(df["question"]))
    unseen["language"] = unseen["question"].map(detect_language)

    df["sources"] = df["final_answer_sources"].map(lambda raw: json.dumps(parse_source_names(raw), ensure_ascii=False))
    df["language"] = df["question"].map(detect_language)
    df["qnorm"] = df["question"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df["answer_key"] = df["final_answer"].map(lambda a: hashlib.md5(" ".join(a.lower().split()).encode()).hexdigest())

    if smoke:
        df = df.sample(n=300, random_state=SEED)
        unseen = unseen.head(20)

    # Same split as the original notebook.
    df = df.drop_duplicates("question_id").drop_duplicates("qnorm").reset_index(drop=True)
    dev, test = train_test_split(df, test_size=0.10, random_state=SEED)
    train, val = train_test_split(dev, test_size=0.10, random_state=SEED)

    tables = {"train": train, "val": val, "test": test, "unseen": unseen}
    tables = {name: table.reset_index(drop=True) for name, table in tables.items()}
    for name, table in tables.items():
        table.to_parquet(files[name], index=False)
    return tables


# ---------------------------------------- Retrieval ----------------------------------------
class Encoder:
    """multilingual-e5-small: mean pooling, then L2 normalisation."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def encode(self, texts, prefix, batch_size=256, max_length=256):
        texts = [prefix + str(t) for t in texts]
        order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths batch faster
        vectors = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=len(texts) <= batch_size):
            rows = order[start : start + batch_size]
            batch = self.tokenizer([texts[i] for i in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vectors[rows] = F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).cpu().numpy()
        return vectors


def build_index(train, encoder, paths):
    """Save the training chunks and their embeddings. Only the training split is indexed."""
    chunks_file, dense_file = paths.index / "chunks.parquet", paths.index / "dense.npy"
    if chunks_file.exists() and dense_file.exists():
        return
    step = CHUNK_WORDS - CHUNK_OVERLAP
    rows = []
    for r in train.itertuples():
        tokens = r.final_answer.split()
        for i in range(0, max(len(tokens) - CHUNK_OVERLAP, 1), step):
            rows.append({"question_id": r.question_id, "answer_id": r.final_answer_id, "answer_key": r.answer_key,
                         "question": r.question, "text": " ".join(tokens[i : i + CHUNK_WORDS]), "sources": r.sources})
    chunks = pd.DataFrame(rows)
    print(len(chunks), "chunks from", len(train), "training answers")
    np.save(dense_file, encoder.encode(chunks["question"] + "\n" + chunks["text"], "passage: "))
    chunks.to_parquet(chunks_file, index=False)  # written last: its presence means the index is complete


def top_rows(scores, k):
    k = min(k, len(scores))
    rows = np.argpartition(-scores, k - 1)[:k]
    return rows[np.argsort(-scores[rows], kind="stable")]


class Retriever:
    """Embedding search plus TF-IDF keyword search, combined with Reciprocal Rank Fusion."""

    def __init__(self, encoder, paths):
        self.encoder = encoder
        self.chunks = pd.read_parquet(paths.index / "chunks.parquet")
        self.dense = np.load(paths.index / "dense.npy")
        if len(self.chunks) != len(self.dense):
            raise RuntimeError("Chunk table and embeddings have different sizes. Delete the index folder and run again.")
        self.vectorizer = TfidfVectorizer(tokenizer=words, token_pattern=None, lowercase=False, sublinear_tf=True, dtype=np.float32)
        self.keyword_matrix = self.vectorizer.fit_transform(self.chunks["question"] + "\n" + self.chunks["text"]).T.tocsr()
        self.text = self.chunks["text"].to_numpy()
        self.groups = {c: self.chunks.groupby(c).indices for c in ("question_id", "answer_id", "answer_key")}
        print("Indexed chunks:", len(self.chunks))

    def own_rows(self, record):
        """Chunks that come from this record's own question or answer. Used to prevent leakage."""
        rows = set()
        for column, value in (("question_id", record.question_id), ("answer_id", record.final_answer_id), ("answer_key", record.answer_key)):
            rows.update(self.groups[column].get(value, []))
        return sorted(int(r) for r in rows)

    def search(self, queries, top_k=TOP_K, exclude=None, hybrid=True, batch_size=256):
        """For each query, return its best chunks as {"row": chunk row, "score": embedding cosine}."""
        queries = list(queries)
        exclude = exclude if exclude is not None else [[] for _ in queries]
        query_vectors = self.encoder.encode(queries, "query: ")
        query_keywords = self.vectorizer.transform(queries)

        results = []
        for start in tqdm(range(0, len(queries), batch_size), desc="Search", disable=len(queries) <= batch_size):
            end = start + batch_size
            dense_scores = query_vectors[start:end] @ self.dense.T
            keyword_scores = (query_keywords[start:end] @ self.keyword_matrix).toarray()
            for i, (dense, keyword) in enumerate(zip(dense_scores, keyword_scores)):
                cosine = dense.copy()
                dense[exclude[start + i]] = -9
                keyword[exclude[start + i]] = -9
                dense_rows = top_rows(dense, max(100, top_k))
                dense_rows = dense_rows[dense[dense_rows] > -9]
                if hybrid:
                    keyword_rows = top_rows(keyword, 50)
                    keyword_rows = keyword_rows[keyword[keyword_rows] > 0]
                    fused = {}
                    for ranking in (dense_rows, keyword_rows):
                        for rank, row in enumerate(ranking, 1):
                            fused[int(row)] = fused.get(int(row), 0) + 1 / (60 + rank)
                    best = sorted(fused, key=lambda row: (-fused[row], row))[:top_k]
                else:
                    best = [int(row) for row in dense_rows[:top_k]]
                results.append([{"row": row, "score": float(cosine[row])} for row in best])
        return results

    def contexts(self, hits):
        return ["\n".join(self.text[h["row"]] for h in row_hits) for row_hits in hits]

    def source_names(self, row_hits):
        names = [n for h in row_hits for n in json.loads(self.chunks["sources"].iat[h["row"]])]
        return list(dict.fromkeys(names))

    def records(self, row_hits, k=10):
        """Unique training question IDs in rank order."""
        return list(dict.fromkeys(self.chunks["question_id"].iat[h["row"]] for h in row_hits))[:k]


# ---------------------------------------- Generation ----------------------------------------
def load_model(paths):
    """Load flan-t5-small with the trained LoRA weights merged in."""
    from peft import PeftModel

    if not (paths.model / "training_info.json").exists():
        raise FileNotFoundError(f"No finished model in {paths.model}. Run 01_train.ipynb first with the same SMOKE_TEST value.")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(paths.model)).merge_and_unload()
    return tokenizer, model.to("cuda").eval()


@torch.inference_mode()
def generate_answers(tokenizer, model, prompts, desc="Generate"):
    """Generate in batches of similar length, and return answers in the original order."""
    start_time = time.time()
    order = np.argsort([len(p) for p in prompts], kind="stable")
    answers = [""] * len(prompts)
    for start in tqdm(range(0, len(prompts), GEN_BATCH_SIZE), desc=desc):
        rows = order[start : start + GEN_BATCH_SIZE]
        batch = tokenizer([prompts[i] for i in rows], padding=True, truncation=True, max_length=MAX_SOURCE_TOKENS, return_tensors="pt").to("cuda")
        output = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, no_repeat_ngram_size=4, use_cache=True)
        for row, answer in zip(rows, tokenizer.batch_decode(output, skip_special_tokens=True)):
            answers[row] = answer.strip()
    print(f"{desc}: {len(prompts) / (time.time() - start_time):.2f} questions per second")
    return answers


def predictions_table(retriever, table, answers, hits):
    """Same columns as the old submission. `matched_image_files` is kept so existing code still works."""
    return pd.DataFrame({
        "question_id": table["question_id"].to_numpy(),
        "question": table["question"].to_numpy(),
        "predicted_answer": answers,
        "predicted_sources": [json.dumps(retriever.source_names(h), ensure_ascii=False) for h in hits],
        "matched_image_files": "[]",
        "detected_language": table["language"].to_numpy(),
        "retrieval_scores": [json.dumps([round(x["score"], 4) for x in h]) for h in hits],
        "confidence": [round(float(np.clip(h[0]["score"], 0, 1)), 4) if h else 0.0 for h in hits],
    })


# ---------------------------------------- Metrics ----------------------------------------
def retrieval_metrics(ranked, relevant):
    """Recall@1/5/10, MRR and nDCG@5 at record level."""
    out = {f"Recall@{k}": np.mean([bool(set(r[:k]) & s) for r, s in zip(ranked, relevant)]) for k in (1, 5, 10)}
    out["MRR"] = np.mean([next((1 / (i + 1) for i, item in enumerate(r) if item in s), 0.0) for r, s in zip(ranked, relevant)])
    ndcg = []
    for r, s in zip(ranked, relevant):
        ideal = sum(1 / np.log2(i + 2) for i in range(min(5, len(s))))
        ndcg.append(sum(1 / np.log2(i + 2) for i, item in enumerate(r[:5]) if item in s) / ideal if ideal else 0.0)
    out["nDCG@5"] = np.mean(ndcg)
    return out


class _RougeTokenizer:
    tokenize = staticmethod(words)


NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*(%|ml|l|litre|liter|g|gm|kg|ha|acre|ppm|mg|q|quintal)?\b", re.I)


def numbers(text):
    return {m.group(1) for m in NUMBER.finditer(str(text))}


def dosages(text):
    return {m.group(1) + m.group(2).lower() for m in NUMBER.finditer(str(text)) if m.group(2)}


def recall(gold, predicted):
    return len(gold & predicted) / len(gold) if gold else np.nan


def bertscore_f1(answers, golds):
    """Per-answer BERTScore. A failure gives NaN, so it can't affect the saved predictions."""
    try:
        from bert_score import score
        # bert-score crashes on an empty string with transformers 5, so give it a placeholder.
        safe = [a if a.strip() else "." for a in answers]
        return score(safe, golds, model_type="bert-base-multilingual-cased", batch_size=64, verbose=False)[2].cpu().numpy()
    except Exception:
        print("BERTScore failed and is reported as NaN.")
        traceback.print_exc()
        return np.full(len(answers), np.nan)


def evaluate_answers(name, answers, sources, test):
    """Return one row per language with text metrics and domain checks."""
    import sacrebleu
    from rouge_score import rouge_scorer

    golds = test["final_answer"].tolist()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=_RougeTokenizer())
    rouge = [scorer.score(g, a) for g, a in zip(golds, answers)]
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    gold_sources = [{norm(n) for n in json.loads(s)} for s in test["sources"]]
    per_answer = pd.DataFrame({
        "ROUGE-1": [r["rouge1"].fmeasure for r in rouge],
        "ROUGE-2": [r["rouge2"].fmeasure for r in rouge],
        "ROUGE-L": [r["rougeL"].fmeasure for r in rouge],
        "BERTScore-F1": bertscore_f1(answers, golds),
        "number_recall": [recall(numbers(g), numbers(a)) for g, a in zip(golds, answers)],
        "dosage_recall": [recall(dosages(g), dosages(a)) for g, a in zip(golds, answers)],
        "source_hit_rate": [bool(g & {norm(n) for n in p}) for g, p in zip(gold_sources, sources)],
        "language_matches_question": [detect_language(a) == q for a, q in zip(answers, test["language"])],
    }, dtype=float)

    rows = []
    languages = test["language"].to_numpy()
    for language in ("overall", "en", "hi", "pa"):
        keep = np.arange(len(test)) if language == "overall" else np.flatnonzero(languages == language)
        if len(keep):
            bleu = sacrebleu.corpus_bleu([answers[i] for i in keep], [[golds[i] for i in keep]]).score
            rows.append({"model": name, "language": language, "n": len(keep), **per_answer.iloc[keep].mean().to_dict(), "SacreBLEU": bleu})
    return rows
