# Agricultural Question Answering: submission guide

This folder holds the code, report, predictions and results for an agricultural question-answering system. Qwen2.5-3B-Instruct is
fine-tuned with QLoRA and given retrieved training answers as references (retrieval-augmented generation), a learned re-ranker
chooses those references, and an exact-match router answers questions that already appear in the training data.

**Start with [`REPORT.pdf`](REPORT.pdf)**, the full evaluation report. [`REPORT.md`](REPORT.md) has the same content as Markdown.

---

## Where each required deliverable is

| Deliverable | Location |
|---|---|
| Kaggle notebook code | [`notebooks/`](notebooks/): source of all 12 notebooks. Executed copies with Kaggle outputs are in [`executed_notebooks/`](executed_notebooks/). Kaggle IDs are in Section 10 of the report. |
| Evaluation report on the training dataset | [`REPORT.pdf`](REPORT.pdf) / [`REPORT.md`](REPORT.md). The raw result files are in [`results/`](results/). |
| Updated CSV of the test queries (Xt) with responses | [`xt_predictions.csv`](xt_predictions.csv) |
| Separate notebook that loads the final checkpoints and generates responses for Xt and the test set | [`notebooks/11_final_inference_full_test.ipynb`](notebooks/11_final_inference_full_test.ipynb), executed copy [`executed_notebooks/11_final_inference_full_test__notebookc8f7682267.ipynb`](executed_notebooks/11_final_inference_full_test__notebookc8f7682267.ipynb). Its test-set output is [`test_predictions.csv`](test_predictions.csv). |

---

## Folder layout

```
submission/
├── README.md                  this file
├── REPORT.pdf / REPORT.md     evaluation report
├── xt_predictions.csv         answers for the 1,000 unseen questions (Xt)
├── test_predictions.csv       final model's answers for the 1,500 hold-out test questions
├── notebooks/                 notebook source code (00-11)
├── executed_notebooks/        the same notebooks with their Kaggle outputs
├── src/                       Python modules the notebooks are built from, plus figure scripts
├── results/                   raw metrics and per-question scores behind every number in the report
└── figures/                   the 9 report figures
```

---

## Prediction files

| File | Rows | Columns | Contents |
|---|---|---|---|
| `xt_predictions.csv` | 1,000 | `question_id, question, final_answer` | Unseen set in its original order. 438 answers come from the router (the question exists in training) and 562 from the final model. No empty answers. |
| `test_predictions.csv` | 1,500 | `question_id, question, reference_answer, final_answer` | Final model's answer to every hold-out test question, next to the human reference answer. |

---

## `notebooks/`: run order and purpose

Notebooks 00-08 build the system, and 09-11 evaluate it. **Notebook 11 is the final inference notebook.**

| # | Notebook | What it does |
|---|---|---|
| 00 | `00_phase0_data_foundation.ipynb` | Normalises questions, groups repeats, picks a medoid answer per group, makes the grouped train/dev/test split (12,825 / 675 / 1,500), builds the evaluation harness. |
| 01 | `01_download_extract_pdfs.ipynb` | Downloads the 7,428 cited source PDFs and extracts per-page text. |
| 02 | `02_quality_and_chunking.ipynb` | Labels page quality and chunks usable pages for retrieval. |
| 03 | `03_retrieval_and_b0.ipynb` | Question retrieval (bge-m3 + BM25, reciprocal-rank fusion), the 10-neighbour files, and the copy baseline B0. |
| 04 | `04_train_qlora.ipynb` | Trains adapter v1 (`nn` with references, `none` without), 4,000 rows. |
| 05 | `05_inference_and_evaluation.ipynb` | First full evaluation (Phase 5). |
| 06 | `06_decoding_sweep.ipynb` | Compares 7 decoding settings (N = 50) and selects setting E. |
| 07 | `07_rerank_neighbours.ipynb` | Learned neighbour re-ranker (25 features, gradient boosting). |
| 08 | `08_retrain_full.ipynb` | Trains adapter v2 on all 12,825 rows with 2-GPU DDP. **This is the final checkpoint.** |
| 09 | `09_final_inference_and_evaluation.ipynb` | All systems on a 300-question development sample, the router, and the unseen-set predictions. |
| 10 | `10_semantic_evaluation.ipynb` | BERTScore and bge-m3 similarity on the same answers (N = 300). |
| 11 | `11_final_inference_full_test.ipynb` | **Final inference and evaluation.** Loads the trained checkpoints, generates answers for Xt and all 1,500 test questions, and scores six systems on every metric. |

`executed_notebooks/` has one file per notebook, named `NN_name__<kaggle-id>.ipynb`, except notebook 02. Its executed copy is not
included.

---

## `results/`: files from notebook 11

| File | Contents |
|---|---|
| `metrics.json` | All aggregate scores (N = 1,500): ROUGE, BLEU, fact-F1, BERTScore, bge-m3 similarity, paired bootstrap intervals, router scores. |
| `per_item_scores.csv` | Every metric for every system on every test question (one row per question). |
| `test_predictions_all_systems.csv` | Answers of all six systems for the 1,500 test questions, with the reference. |
| `xt_predictions_detailed.csv` | `xt_predictions.csv` plus a `source` column (`router` or the model). |
| `xt_predictions_with_baseline.csv` | `xt_predictions.csv` plus a `baseline_answer` column (router, then the top re-ranked copied answer). |
| `run_config.json` | Model, adapter and decoding settings of the final run. |
| `notebook11_kaggle_log.json` | Kaggle execution log of notebook 11. |

The six systems compared are:
- copy the nearest training answer, original order (B0a)
- copy the nearest training answer, re-ranked (B0a)
- zero-shot base model (B1)
- fine-tuned without retrieval (B2)
- fine-tuned v1 with retrieval (P1)
- the **final model** (P2: adapter v2 with re-ranked references)

---

## `src/`: source modules

| Module | Purpose |
|---|---|
| `phase0_prepare.py`, `sources.py` | Data foundation; parsing the cited sources. |
| `eval_harness.py` | The single evaluation harness used by every system. |
| `build_manifest.py`, `pdf_pipeline.py`, `chunking.py`, `phase1_worker.py` | PDF download, text extraction, quality labels, chunking. |
| `bm25.py`, `encoders.py`, `retrieval.py` | Sparse and dense retrieval, reciprocal-rank fusion. |
| `rerank.py` | Learned neighbour re-ranker. |
| `prompts.py` | Training and inference prompt construction. |
| `train_qlora.py` | QLoRA fine-tuning (loss on answer tokens only). |
| `infer.py`, `decode_sweep.py` | Generation workers (one process per GPU). |
| `diag_retrieval.py`, `diag_generation.py`, `diag_gate.py`, `step1_postprocess.py` | CPU diagnostics on saved outputs (Sections 4.2, 4.6, 4.11 of the report). |
| `make_sweep_payload.py`, `make_semantic_payload.py` | Input bundles for notebooks 06 and 10. |
| `build_notebooks.py`, `build_nb10.py`, `build_nb11.py` | Generate the notebooks in `notebooks/` from these modules. |
| `make_figures.py`, `make_diagrams.py` | Draw the report charts from the result files, and render the diagrams. |

---

## `figures/`

| File | Report figure |
|---|---|
| `fig1_architecture_final.png` | Figure 1: system architecture |
| `fig2_timeline_final.png` | Figure 2: project timeline and decisions |
| `fig3_training_example_final.png` | Figure 3: one training example and QLoRA |
| `fig4_decoding.png` | Figure 4: decoding sweep |
| `fig5_reranker_final.png` | Figure 5: neighbour re-ranker |
| `fig6_dev_loss.png` | Figure 6: dev loss of adapter v2 |
| `fig7_quartiles.png` | Figure 7: final model vs copy by question similarity |
| `fig8_bootstrap.png` | Figure 8: paired bootstrap differences |
| `fig9_lexical_vs_semantic.png` | Figure 9: all systems on ROUGE-L and BERTScore |

---

## Headline results (all 1,500 hold-out questions; notebook 11)

| System | ROUGE-L | BERTScore F1 | bge-m3 similarity |
|---|---|---|---|
| Copy nearest answer (original order) | 0.2171 | 0.8425 | 0.777 |
| Copy nearest answer (re-ranked) | **0.2322** | 0.8454 | 0.788 |
| Zero-shot base model (B1) | 0.1371 | 0.8259 | 0.821 |
| Fine-tuned, no retrieval (B2) | 0.1653 | 0.8325 | 0.821 |
| Fine-tuned v1 + retrieval (P1) | 0.2040 | 0.8464 | 0.838 |
| **Final model (P2)** | 0.2125 | **0.8499** | **0.846** |

The final model is the best system on the meaning-based metrics. Copying the re-ranked nearest answer is the best on word overlap.
Full results, confidence intervals and the reasoning behind each design decision are in the report (Sections 4.13 and 5).
