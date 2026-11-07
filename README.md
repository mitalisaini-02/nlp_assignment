# Fast Agricultural QA Pipeline

This pipeline answers farmer questions using retrieved agricultural knowledge and a fine-tuned `flan-t5-small`. It replaces the single notebook `nlp_final_assignment (1).ipynb` with two Google Colab notebooks, one for training and one for inference.

> **Status:** this code has only had static checks. It has never been run, so do the smoke test in step 3 before the full run.

## What changed from the old notebook

| Part | Old notebook | This pipeline |
|---|---|---|
| Embeddings | `BAAI/bge-m3` | `intfloat/multilingual-e5-small` |
| Answer generator | `Qwen/Qwen2.5-3B-Instruct` | `google/flan-t5-small` |
| Search | FAISS, BGE sparse vectors and BM25 | Embedding search with NumPy, plus TF-IDF keywords |
| Reranker | `bge-reranker-v2-m3` | Removed |
| Images | SigLIP2 and BLIP | Removed; the CSVs don't link questions to images |
| Language detection | Lingua on questions and answers | Unicode script rule on questions only |
| Generation | 1 question at a time, up to 768 new tokens | 32 per batch, up to 192 new tokens |

## Files

| File | Purpose |
|---|---|
| `agri_common.py` | Shared settings and code |
| `01_train.ipynb` | Prepares data, builds the index, evaluates retrieval, trains the model |
| `02_inference.ipynb` | Generates answers, saves the submission, computes metrics |

## How to run

### 1. Upload to Google Drive

Create this layout. The CSV names must match exactly:

```text
MyDrive/agricultural_qa/
├── agri_common.py
└── data/
    ├── training_set_15000.csv
    └── unseen_set_1000.csv
```

Upload both notebooks to Colab with *File > Upload notebook*.

### 2. Select the GPU

In each notebook, choose *Runtime > Change runtime type > T4 GPU*.

### 3. Run the smoke test

This uses 300 rows, 1 epoch and 20 questions, and writes to a separate `fast_v1_smoke/` folder.

1. In `01_train.ipynb`, set `SMOKE_TEST = True` and choose *Runtime > Run all*.
2. Open `02_inference.ipynb` in a **fresh runtime**, set `SMOKE_TEST = True`, and run all.

The answers will be poor at this size. You're only checking that both notebooks finish without errors.

### 4. Run the full pipeline

1. In `01_train.ipynb`, set `SMOKE_TEST = False` and run all. Wait for `Saved. Open 02_inference.ipynb in a fresh runtime.`
2. Choose *Runtime > Disconnect and delete runtime*.
3. In `02_inference.ipynb`, set `SMOKE_TEST = False` and run all.
4. Download `submission_predictions.csv` from the `outputs/` folder. The notebook prints the exact location.

## Workflow

```text
training_set_15000.csv     unseen_set_1000.csv
          └──────────┬──────────┘
                     ▼
              01_train.ipynb
   1. Install packages and mount Drive
   2. Load data: parse CSVs, dedupe, split 81 / 9 / 10
   3. Build the index from training answers only
   4. Evaluate retrieval: dense baseline vs hybrid
   5. Build prompts, excluding each record's own answer
   6. Train flan-t5-small with LoRA
   7. Plot the loss curve and save to Drive
                     │
                     ▼  fast_v1/<run ID>/ on Drive
              02_inference.ipynb
   1. Install packages and mount Drive
   2. Load the saved data
   3. Retrieve context for every question
   4. Generate answers and save predictions
   5. Compute metrics
   6. Print error analysis and save to Drive
```

### Notebook 1: `01_train.ipynb`

| Step | What happens | Saved in the run folder |
|---|---|---|
| Load data | Reads each CSV once, parses sources, sets language, removes duplicates, splits the data | `data/*.parquet` |
| Build index | Splits answers into 120-word chunks with a 20-word overlap, then embeds them | `index/chunks.parquet`, `index/dense.npy` |
| Evaluate retrieval | Compares embedding search with hybrid search using Recall@1/5/10, MRR and nDCG@5 | `outputs/retrieval_results.csv` |
| Build prompts | Searches for each training question with its own answer excluded | `prompts/*.parquet` |
| Train | Trains LoRA with batch size 16 for 3 epochs, with early stopping and a time limit | `checkpoints/`, `model/`, `outputs/training_history.csv` |

Hybrid search combines embedding and TF-IDF keyword results with Reciprocal Rank Fusion. Each prompt looks like this:

```text
Answer the farmer's question using the context. Preserve scientific names, chemical names, dosages, numbers and units. Do not invent recommendations.
question: {question}
context: {top 3 retrieved chunks}
```

The context comes last, so cutting a long prompt at 512 tokens can only shorten the context, not the question.

### Notebook 2: `02_inference.ipynb`

| Step | What happens | Saved in the run folder |
|---|---|---|
| Retrieve | Runs hybrid search for every test and unseen question | None |
| Generate | Loads the trained model and answers in batches of 32 | `outputs/test_predictions.csv`, `outputs/submission_predictions.csv` |
| Metrics | Computes ROUGE, SacreBLEU, BERTScore, number and dosage recall, source hits and output language | `outputs/evaluation_results.csv` |
| Error analysis | Prints wrong output language, dosage errors and answers with no gold source | None |

Both prediction files use these columns: `question_id`, `question`, `predicted_answer`, `predicted_sources`, `matched_image_files` (always `[]`, kept for compatibility), `detected_language`, `retrieval_scores`, `confidence`. `confidence` is the top result's cosine similarity, not the chance that the answer is correct.

## Expected runtime

These are estimates for a T4 GPU; nothing has been timed yet. Actual times are saved in `outputs/timings.json`.

| Notebook | Stage | Estimate |
|---|---|---|
| `01_train.ipynb` | Data, index, retrieval evaluation, prompts | 10–20 min |
| `01_train.ipynb` | Training | 15–35 min |
| `02_inference.ipynb` | Retrieval and generation for 2,439 questions | 5–15 min |
| `02_inference.ipynb` | Metrics | 5–10 min |

Training stops by itself at 55 minutes of total notebook time and keeps the best checkpoint. If the earlier steps use up that time, training still gets at least 5 minutes. `model/training_info.json` records whether the limit was reached.

## If the runtime disconnects

- **`01_train.ipynb`:** run all again. Finished steps are reused and training continues from its last checkpoint, saved every 200 steps.
- **`02_inference.ipynb`:** run all again. Retrieval and generation start over because they don't save partial progress.

Keep the same `SMOKE_TEST` value when you run again.

## Settings

All settings are at the top of `agri_common.py`, so both notebooks always use the same values. Upload the file again after any change, then restart the runtime.

| Setting | Default | Use |
|---|---|---|
| `TRAIN_BATCH_SIZE` | `16` | Lower it if training runs out of GPU memory |
| `GEN_BATCH_SIZE` | `32` | Lower it if generation runs out of GPU memory |
| `MAX_NEW_TOKENS` | `192` | Allow longer answers, at the cost of speed |
| `TRAIN_BUDGET_MINUTES` | `55` | Change the training time limit |

Changing a chunk, prompt, model or training setting creates a new run folder and rebuilds everything, so old files can't be mixed in by accident. The last three settings above don't trigger a rebuild.

## Troubleshooting

| Message | Fix |
|---|---|
| `No module named 'agri_common'` | Put the file directly inside `MyDrive/agricultural_qa/` |
| `Missing ... training_set_15000.csv` | Check the folder and the exact file name |
| `No GPU found` | Select the T4 GPU and run again |
| `No finished model` | Finish `01_train.ipynb` with the same `SMOKE_TEST` value and settings |
| `CUDA out of memory` | Lower the batch size, upload the file again and restart the runtime |
| `BERTScore failed` | Predictions and other metrics are safe; the score is reported as `NaN` |

## Known limitations

- **Answers are shorter and less detailed:** `flan-t5-small` is a small model limited to 192 new tokens, while the median correct answer is 412 words. Expect lower scores than a larger model could reach.
- **Scores can't be compared with the old notebook:** the embedding model, retrieval metric and generator all changed.
- **Hindi and Punjabi are weak:** `flan-t5-small` handles these scripts poorly, and answers will often be English or low quality. The training set has only 19 such questions, and the unseen set has 7.
- **Old indexes and checkpoints aren't used:** this pipeline builds its own. The old index needs BGE-M3 to search it, and the old adapter only works with Qwen.
- **Unseen questions overlap with training:** 431 of the 1,000 unseen questions appear word for word in the training file. The pipeline doesn't copy stored answers, but it can retrieve them as context, so scores on these questions may look better than on truly new ones.
- **No image support:** the pipeline answers from text only.
