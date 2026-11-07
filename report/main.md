\documentclass[11pt,a4paper]{article}

\usepackage[margin=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage{caption}
\usepackage{authblk}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!55!black,
  urlcolor=blue!55!black,
  citecolor=blue!55!black,
  pdftitle={Agricultural Question Answering Model: Evaluation Report},
  pdfauthor={Project Team}
}

\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.18}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\metric}[1]{\texttt{#1}}
\newcommand{\todo}[1]{\textcolor{red}{[ #1 ]}}

\title{\textbf{Agricultural Question Answering Model}\\
       \large Development and Evaluation Report}
\author[1]{Lakshay\thanks{Roll No.: 2023AIB1010}}
\author[1]{Kumar Vedant\thanks{Roll No.: 2023AIB1009}}
\author[1]{Mitali Saini\thanks{Roll No.: 2023CSB1134}}
\author[1]{Shivam Goyal\thanks{Roll No.: 2023AIB1016}}
\affil[1]{Indian Institute of Technology Ropar, Rupnagar, Punjab, India}
\renewcommand\Authands{ and }
\date{November 5, 2026}
\setlength{\affilsep}{0.5em}
\renewcommand\Authfont{\normalsize}
\renewcommand\Affilfont{\small}

\begin{document}
\maketitle

\begin{abstract}
This report describes the implemented agricultural question-answering (Q\&A)
system and its evaluation on the held-out portion of the supplied training
dataset. The system uses multilingual-e5-small for dense text embeddings,
TF--IDF for keyword retrieval, Reciprocal Rank Fusion (RRF) to combine
retrieval rankings, and a LoRA fine-tuned FLAN-T5-small model to generate
answers from retrieved context. The cleaned dataset contains 14,390 unique
questions after duplicate removal and is divided into 11,655 training,
1,296 validation, and 1,439 test examples. The final model was evaluated
using text-overlap, semantic, numerical-preservation, source, language, and
retrieval metrics. On the 1,439-question test set, the generated answers
obtained ROUGE-1 of 0.2350, ROUGE-2 of 0.0688, ROUGE-L of 0.1364,
BERTScore-F1 of 0.6723, and SacreBLEU of 0.4830. Numerical and dosage
recall were 0.0976 and 0.0305, respectively. The results show that the
pipeline successfully produces outputs for the complete test and unseen
query sets, but answer completeness, numerical fidelity, and Punjabi
generation remain important limitations.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}

The objective of this work is to develop a Question-and-Answer model that
takes an agricultural question as input and generates a corresponding
response using the supplied question-answer training data. The supplied
material also contains response-source information and crop-image samples.
The implemented version described in this report is a \textbf{text-only
retrieval-augmented generation system}; crop images are not used by this
architecture.

The system has two main stages:
\begin{enumerate}[leftmargin=*]
  \item \textbf{Offline preparation and training:} clean and split the
  question-answer data, construct a searchable index from training answers,
  create retrieval-grounded prompts, and fine-tune a sequence-to-sequence
  generator using LoRA.
  \item \textbf{Inference and evaluation:} retrieve context for each query,
  generate an answer with the trained model, save predictions, and evaluate
  the test-set outputs.
\end{enumerate}

This report documents the new architecture implemented in
\texttt{agri\_common.py}, \texttt{01\_train.ipynb}, and
\texttt{02\_inference.ipynb}. It reports the new architecture's results
without presenting results from an earlier architecture or a separate
baseline system.

\section{Dataset and Preprocessing}

\subsection{Dataset fields}

The training data contains questions and their reference answers. The
implementation also reads question identifiers, answer identifiers, and
the \texttt{final\_answer\_sources} field. Source names are parsed and
retained as metadata for retrieval and source-related evaluation.

The unseen-query file contains 1,000 questions. The inference notebook
generates one response for each of these questions.

\subsection{Cleaning and duplicate handling}

Before splitting, the implementation:
\begin{itemize}[leftmargin=*]
  \item reads the training and unseen CSV files as strings;
  \item parses source names from the source metadata;
  \item assigns a language label from the question script (Gurmukhi is
  labelled Punjabi, Devanagari is labelled Hindi, and other text is
  labelled English);
  \item creates a normalized question string by lowercasing, replacing
  non-word character sequences with spaces, and trimming whitespace;
  \item removes duplicate \texttt{question\_id} values and duplicate
  normalized questions.
\end{itemize}

After these operations, the training table used by the notebooks contains
14,390 examples. The code uses a fixed random seed of 42 for splitting.

\subsection{Train, validation, and test split}

The code first reserves 10\% of the deduplicated data for testing. It then
splits the remaining 90\% into training and validation subsets, using
10\% of that remainder for validation. The resulting proportions are
therefore approximately 81\% training, 9\% validation, and 10\% testing.

\begin{table}[H]
\centering
\caption{Dataset split used in the executed run.}
\begin{tabular}{lrr}
\toprule
\textbf{Subset} & \textbf{Number of examples} & \textbf{Share} \\
\midrule
Training & 11,655 & 81\% \\
Validation & 1,296 & 9\% \\
Test & 1,439 & 10\% \\
\midrule
Total after deduplication & 14,390 & 100\% \\
Unseen queries (separate file) & 1,000 & -- \\
\bottomrule
\end{tabular}
\end{table}

The training-prompt construction uses all 11,655 training examples.
For validation prompt construction, the notebook samples 300 examples
from the validation subset using seed 42. The remaining validation
examples are not used in the recorded prompt set for training-time
evaluation. The held-out test subset is kept separate from training and
is used for final metric calculation.

\subsection{Language distribution}

The saved test evaluation contains 1,436 English questions and 3 Punjabi
questions. No Hindi test examples appear in the saved metric table.
Consequently, the overall scores are dominated by English examples, and
the Punjabi results are based on a very small sample.

\section{System Architecture}

\subsection{Architecture overview}

For a query \(q\), the system retrieves text chunks from the training
answer corpus and places the retrieved context into an instruction prompt.
The fine-tuned generator then produces the response:

\[
q \longrightarrow
\text{Embedding and keyword retrieval}
\longrightarrow
\text{RRF ranking}
\longrightarrow
\text{Top-3 context}
\longrightarrow
\text{FLAN-T5-small + LoRA}
\longrightarrow \hat{a}.
\]

\subsection{Text chunking and index construction}

Only the training split is indexed. Each training answer is divided into
chunks of 120 words with an overlap of 20 words. Each chunk retains its
question identifier, answer identifier, normalized-answer key, original
question, and source-name metadata.

The executed inference notebook reports an index containing 60,554
chunks. Chunk embeddings are generated using
\texttt{intfloat/multilingual-e5-small}. The encoder applies mean pooling
over non-padding token representations and L2-normalizes the resulting
vectors. Query and passage prefixes are used when encoding questions and
indexed text, respectively.

\subsection{Hybrid retrieval}

The retriever combines two retrieval signals:

\begin{enumerate}[leftmargin=*]
  \item \textbf{Dense retrieval:} cosine similarity between normalized
  multilingual-e5-small query embeddings and chunk embeddings.
  \item \textbf{Keyword retrieval:} TF--IDF similarity using a tokenizer
  that retains word tokens from the supported scripts.
\end{enumerate}

The two ranked lists are combined using Reciprocal Rank Fusion. For a
candidate \(d\), the fusion score is:

\[
\operatorname{RRF}(d)
=
\sum_{r \in \{\text{dense},\text{keyword}\}}
\frac{1}{60+\operatorname{rank}_{r}(d)}.
\]

The constant \(60\) is the RRF rank-offset parameter used in the code.
The top three chunks are selected to construct the context supplied to
the generator.

During training-prompt construction, chunks belonging to the current
training record's own question, answer identifier, or normalized answer
are excluded. The notebook also checks that a training prompt does not
contain a chunk from its own answer. This is intended to prevent direct
answer leakage into the training context.

\subsection{Prompt construction}

The prompt contains an instruction, the question, and the retrieved
context. The instruction asks the model to answer using the context and
to preserve scientific names, chemical names, dosages, numbers, and units,
without inventing recommendations.

The source sequence is truncated to at most 512 tokens. The target answer
is truncated to at most 256 tokens during training. These limits constrain
the amount of context and reference-answer text available to the model.

\subsection{Answer generation model}

The generator is \texttt{google/flan-t5-small}, a sequence-to-sequence
instruction-tuned model. It is adapted using Parameter-Efficient Fine
Tuning (PEFT) with LoRA. The configuration in the training notebook is:

\begin{table}[H]
\centering
\caption{Generator and LoRA configuration.}
\begin{tabularx}{\linewidth}{lY}
\toprule
\textbf{Setting} & \textbf{Value} \\
\midrule
Base generator & \texttt{google/flan-t5-small} \\
Fine-tuning method & LoRA (PEFT) \\
Task type & Sequence-to-sequence language modelling \\
LoRA rank \(r\) & 16 \\
LoRA alpha & 32 \\
LoRA dropout & 0.05 \\
Target modules & \texttt{q}, \texttt{k}, \texttt{v}, \texttt{o} \\
Training batch size & 16 per device \\
Learning rate & \(5\times 10^{-4}\) \\
Learning-rate scheduler & Cosine \\
Configured epochs & 3 \\
Maximum source length & 512 tokens \\
Maximum target length & 256 tokens \\
Inference maximum new tokens & 192 \\
Inference decoding & Greedy decoding (\texttt{num\_beams=1}) \\
\bottomrule
\end{tabularx}
\end{table}

The training code uses the sequence-to-sequence trainer, a
sequence-to-sequence data collator, early stopping based on validation
loss, and checkpoint selection based on the lowest validation loss.
Training is also subject to a time budget configured in the shared
implementation.

\section{Training Procedure}

The executed training history contains 2,187 optimization steps. Training
loss values were logged every 25 steps and validation loss was recorded
at evaluation intervals. The first logged training loss was 3.6082 and
the last logged training loss was 3.0813. The first recorded validation
loss was 3.0668 and the final recorded validation loss was 2.8177; the
lowest recorded validation loss was also 2.8177.

\begin{table}[H]
\centering
\caption{Selected values from the saved training history.}
\begin{tabular}{lrr}
\toprule
\textbf{Quantity} & \textbf{First recorded} & \textbf{Last recorded} \\
\midrule
Training loss & 3.6082 & 3.0813 \\
Validation loss & 3.0668 & 2.8177 \\
\bottomrule
\end{tabular}
\end{table}

The decreasing recorded losses indicate that optimization reduced the
training objective and that validation loss also improved over the
recorded evaluations. Loss reduction alone does not establish that the
generated answers are factually correct or complete; this is why the
separate answer-quality metrics and error analysis are necessary.

The notebook output records a Tesla T4 GPU. The saved stage timings are
summarized below.

\begin{table}[H]
\centering
\caption{Recorded stage timings from \texttt{timings.json}.}
\begin{tabular}{lr}
\toprule
\textbf{Stage} & \textbf{Time (seconds)} \\
\midrule
Load data & 4.2 \\
Build index & 171.6 \\
Evaluate retrieval & 79.8 \\
Build prompts & 43.1 \\
Train & 2,260.1 \\
Retrieve (inference) & 50.2 \\
Generate & 284.9 \\
Metrics & 341.4 \\
\bottomrule
\end{tabular}
\end{table}

\section{Evaluation Methodology}

\subsection{Test-set generation}

The inference notebook loads the same data split and the trained LoRA
adaptation, retrieves context for each test question, and generates an
answer. The saved \texttt{test\_predictions.csv} contains 1,439 rows,
with no empty predicted-answer fields. The unseen-query output contains
1,000 rows, also with no empty predicted-answer fields.

The unseen set is used for producing submission responses. Since
reference answers for this set are not included in the evaluated output,
the answer-quality metrics reported below are calculated on the labelled
test set, not on the unseen set.

\subsection{Answer-quality metrics}

The implementation calculates the following metrics:

\begin{itemize}[leftmargin=*]
  \item \textbf{ROUGE-1, ROUGE-2, and ROUGE-L:} lexical overlap between
  generated and reference answers, measured using F-measure.
  \item \textbf{BERTScore-F1:} contextual semantic similarity between
  generated and reference answers, using
  \texttt{bert-base-multilingual-cased}.
  \item \textbf{SacreBLEU:} corpus-level n-gram translation-style
  similarity between generated answers and reference answers.
  \item \textbf{Number recall:} proportion of distinct numeric values
  present in the reference answer that are also found in the generated
  answer.
  \item \textbf{Dosage recall:} proportion of reference numeric
  quantity-and-unit matches detected by the implementation's dosage
  regular expression that are also present in the generated answer.
  \item \textbf{Source hit rate:} proportion of examples for which at
  least one retrieved source name matches a normalized source name in
  the reference metadata.
  \item \textbf{Language match:} proportion of generated answers whose
  detected script-based language label matches the question's label.
\end{itemize}

These metrics measure different properties and should not be interpreted
as interchangeable measures of correctness. In particular, lexical
overlap does not establish factual correctness, and number/dosage recall
does not verify that a retained value is used in the correct agronomic
context.

\subsection{Retrieval evaluation}

The saved retrieval evaluation treats a training record as relevant when
the cosine similarity between its reference answer embedding and the
test reference answer embedding is at least 0.80. It reports
Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR), and nDCG@5.
This is a semantic-similarity-based proxy for relevance, not a manually
annotated judgement that the retrieved record answers the test question.

\section{Results}

\subsection{Overall test-set answer quality}

Table~\ref{tab:answer-results} reports the saved metrics for the new
architecture on all 1,439 test questions.

\begin{table}[H]
\centering
\caption{Final architecture: overall test-set answer-quality metrics.}
\label{tab:answer-results}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Score} \\
\midrule
ROUGE-1 & 0.2350 \\
ROUGE-2 & 0.0688 \\
ROUGE-L & 0.1364 \\
BERTScore-F1 & 0.6723 \\
SacreBLEU & 0.4830 \\
Number recall & 0.0976 \\
Dosage recall & 0.0305 \\
Source hit rate & 0.1862 \\
Language matches question & 0.9979 \\
\bottomrule
\end{tabular}
\end{table}

The ROUGE and SacreBLEU scores indicate limited lexical overlap between
the generated responses and the reference answers. BERTScore-F1 of
0.6723 indicates some semantic similarity, but it should not be read as
a probability of correctness. Number recall of 0.0976 and dosage recall
of 0.0305 are particularly low. These results show that the model often
fails to preserve numerical details and quantity-unit expressions that
appear in the reference answers.

The source hit rate is 0.1862. Under the implemented exact normalized
source-name matching procedure, at least one retrieved source name
matches a reference source name for approximately 18.62\% of test
examples. This does not by itself measure whether the generated answer
is supported by the retrieved source.

\subsection{Language-wise results}

\begin{table}[H]
\centering
\caption{Final architecture: answer-quality metrics by detected question language.}
\label{tab:language-results}
\small
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{English (\(n=1436\))} &
\textbf{Punjabi (\(n=3\))} \\
\midrule
ROUGE-1 & 0.2354 & 0.0697 \\
ROUGE-2 & 0.0689 & 0.0209 \\
ROUGE-L & 0.1366 & 0.0439 \\
BERTScore-F1 & 0.6726 & 0.5353 \\
SacreBLEU & 0.4836 & 0.1108 \\
Number recall & 0.0977 & 0.0667 \\
Dosage recall & 0.0306 & 0.0000 \\
Source hit rate & 0.1866 & 0.0000 \\
Language matches question & 1.0000 & 0.0000 \\
\bottomrule
\end{tabular}
\end{table}

All 1,436 English test questions have generated answers classified as
English by the script-based detector. None of the three Punjabi
questions has a generated answer classified as Punjabi. The Punjabi
sample is too small to support a broad estimate of multilingual
performance, but the observed outputs identify Punjabi answer generation
as a clear failure case in this run.

No Hindi test-set row is present in the saved evaluation table, so Hindi
test performance cannot be assessed from these results.

\subsection{Retrieval results}

\begin{table}[H]
\centering
\caption{Final architecture: retrieval metrics using the 0.80 answer-embedding similarity relevance rule.}
\label{tab:retrieval-results}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Score} \\
\midrule
Recall@1 & 1.0000 \\
Recall@5 & 1.0000 \\
Recall@10 & 1.0000 \\
MRR & 1.0000 \\
nDCG@5 & 0.9995 \\
\bottomrule
\end{tabular}
\end{table}

The recorded retrieval metrics are close to 1.0. However, the relevance
rule marks a training answer as relevant based on answer-embedding
cosine similarity of at least 0.80. Multiple training answers can
therefore qualify as relevant for a test example, even when they are not
the exact answer to its question. The retrieval scores should consequently
be interpreted as performance under this proxy relevance definition,
not as proof of perfect question-answer retrieval.

\subsection{Prediction-file checks}

The saved output files were inspected for row counts and empty answers.

\begin{table}[H]
\centering
\caption{Prediction output checks.}
\begin{tabular}{lrr}
\toprule
\textbf{Check} & \textbf{Test output} & \textbf{Unseen output} \\
\midrule
Rows generated & 1,439 & 1,000 \\
Empty predicted answers & 0 & 0 \\
Columns in output CSV & 8 & 8 \\
\bottomrule
\end{tabular}
\end{table}

The inference notebook also reports that 435 unseen questions appear
word-for-word in the original training CSV. The implementation does not
simply copy the stored answer for these questions; it runs retrieval and
generation. Nevertheless, overlap means the unseen set is not entirely
composed of novel question wording, and this should be considered when
interpreting any subsequent external evaluation.

\section{Discussion and Limitations}

The implemented pipeline demonstrates an end-to-end workflow that
combines dense retrieval, keyword retrieval, rank fusion, and
parameter-efficient answer generation. It produces a prediction for
every row in both the held-out test set and the unseen-query set.

The saved evaluation also identifies several limitations:

\begin{enumerate}[leftmargin=*]
  \item \textbf{Answer overlap and completeness:} the ROUGE and SacreBLEU
  values are low, indicating that generated answers differ substantially
  in wording and/or content from the reference answers.
  \item \textbf{Numerical fidelity:} number recall and dosage recall are
  low. This is a critical limitation for agricultural guidance because
  quantities, concentrations, application rates, and units can be
  important.
  \item \textbf{Short generation limit:} inference is capped at 192 new
  tokens. The shared code and README note that the median reference
  answer is much longer than this generation budget. The cap can prevent
  the model from returning all details in a long reference answer.
  \item \textbf{Small generator:} FLAN-T5-small is a compact generator.
  The current results do not establish that it can reliably synthesize
  detailed agricultural advice from retrieved passages.
  \item \textbf{Punjabi generation:} all three Punjabi test answers fail
  the language-match check. The current language detector is also
  script-based and treats text outside the Gurmukhi and Devanagari
  ranges as English; it is not a general-purpose language identifier.
  \item \textbf{Text-only context:} crop-image samples are not integrated
  into this architecture. The model therefore cannot use image evidence
  to answer image-dependent questions.
  \item \textbf{Retrieval metric limitation:} retrieval relevance is
  defined through answer-embedding similarity rather than manual
  question-specific relevance annotations.
  \item \textbf{Unseen-set overlap:} 435 unseen questions are reported
  as exact question-text matches with the original training CSV. This
  reduces the novelty of part of the unseen set.
\end{enumerate}

The reported metrics are automatic measures. They do not replace expert
review of agricultural safety, factual correctness, or whether a
recommendation is appropriate for a particular crop, region, pest, or
disease. The system should not be treated as a substitute for qualified
agricultural advice.

\section{Conclusion}

The project implements a text-based agricultural Q\&A pipeline using
multilingual-e5-small embeddings, TF--IDF keyword retrieval, RRF ranking,
and a LoRA-adapted FLAN-T5-small generator. The data was deduplicated and
split into 11,655 training, 1,296 validation, and 1,439 test examples.
The inference notebook generated non-empty predictions for all 1,439
test questions and all 1,000 unseen questions.

For the new architecture, the overall test results were ROUGE-1 0.2350,
ROUGE-2 0.0688, ROUGE-L 0.1364, BERTScore-F1 0.6723, and SacreBLEU
0.4830. The numerical and dosage recall values were 0.0976 and 0.0305.
Retrieval metrics were near 1.0 under the implemented
answer-embedding-similarity relevance rule, but that rule is only a
proxy and should be interpreted cautiously.

The main conclusion is that the implementation completes the end-to-end
Q\&A workflow and produces the required prediction files, while the
measured answer quality still has substantial limitations. In particular,
numerical and dosage preservation, Punjabi generation, long-answer
completeness, and stronger retrieval relevance validation require further
work before the system can be considered reliable for practical
agricultural use.

\section*{Reproducibility and Submitted Artifacts}

The repository contains the following implementation and result files:
\begin{itemize}[leftmargin=*]
  \item \texttt{notebooks/01\_train.ipynb}: data preparation, index
  construction, retrieval evaluation, prompt creation, and LoRA training.
  \item \texttt{notebooks/02\_inference.ipynb}: checkpoint loading,
  test/unseen inference, prediction export, metrics, and error analysis.
  \item \texttt{agri\_common.py}: shared preprocessing, retrieval,
  generation, and evaluation functions.
  \item \texttt{outputs/evaluation\_results.csv}: saved answer-quality
  metrics.
  \item \texttt{outputs/retrieval\_results.csv}: saved retrieval metrics.
  \item \texttt{outputs/training\_history.csv}: saved training and
  validation loss history.
  \item \texttt{outputs/test\_predictions.csv}: held-out test predictions.
  \item \texttt{outputs/submission\_predictions.csv}: unseen-query
  predictions.
\end{itemize}

The repository archive reviewed for this report does not include the
final model checkpoint directory. The inference notebook expects the
trained model and its LoRA files to be available in the configured
Google Drive run directory. The checkpoint must therefore be preserved
and made available with the final submission if independent inference
reproduction is required.

\end{document}