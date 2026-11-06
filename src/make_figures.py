"""Report charts (Figures 5-9), drawn straight from the saved result files - no number is typed in by hand.

Inputs : notebook 11 metrics.json (N = 1,500), notebook 10 semantic_metrics.json (quartiles, N = 300),
         notebook 08 train_summary.json (dev loss), notebook 06 sweep_results.json (decoding sweep, N = 50)
Output : figures/fig5..fig9 (.png, 200 dpi, and .pdf)

Run:  C:/nlp_data/venv/Scripts/python.exe -I src/make_figures.py
Pure ASCII source on purpose.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
K = os.environ.get("KAGGLE_OUT", "C:/nlp_data/kaggle_out")
M11 = json.load(open(K + "/p11/metrics.json"))
SEM = json.load(open(K + "/p10/semantic_metrics.json"))
TRAIN = json.load(open(K + "/p8/adapters/nn/train_summary.json"))
SWEEP = json.load(open(K + "/p6/sweep_results.json"))
assert M11["status"] == "complete" and M11["test_scored"] == 1500

# validated categorical slots (dataviz reference palette, light mode) + neutral ink
FINAL, COPY = "#2a78d6", "#eb6834"
NEUTRAL, INK, INK2, GRID = "#a3a29b", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

SYS = [  # (key in metrics.json, display name)
    ("B0a copy nearest answer (original order)", "Copy nearest answer (original order)"),
    ("B0a copy nearest answer (re-ranked)", "Copy nearest answer (re-ranked)"),
    ("B1 zero-shot base model", "Zero-shot base model"),
    ("B2 fine-tuned v1, no retrieval", "Fine-tuned, no retrieval"),
    ("P1 fine-tuned v1 + retrieval (Phase 4)", "Fine-tuned v1 + retrieval"),
    ("P2 FINAL fine-tuned v2 + re-ranked retrieval", "Final model"),
]


def save(fig, name, note, y=-0.02):
    fig.text(0.0, y, note, fontsize=7.5, color=INK2, ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, name + "." + ext))
    plt.close(fig)
    print("wrote", name)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig5():
    """Dot plot: each system on ROUGE-L and BERTScore F1. Dots, not bars, because the BERTScore axis cannot start at 0."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    names = [d for _, d in SYS][::-1]
    for ax, metric, title, fmt in ((axes[0], "rougeL", "ROUGE-L (word overlap)", "%.4f"),
                                   (axes[1], "bertscore_f1", "BERTScore F1 (meaning)", "%.4f")):
        vals = [M11["systems"][k][metric] for k, _ in SYS][::-1]
        for y, (n, v) in enumerate(zip(names, vals)):
            final = n == "Final model"
            ax.hlines(y, min(vals) - (max(vals) - min(vals)) * 0.08, v, color=GRID, linewidth=1)
            ax.plot(v, y, "o", ms=8, color=FINAL if final else NEUTRAL, mec="white", mew=1.5, zorder=3)
            ax.annotate(fmt % v, (v, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK if final else INK2, fontweight="bold" if final else "normal")
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.22)
        ax.set_title(title)
        xgrid(ax)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)), names)
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == "Final model":
            lab.set_fontweight("bold"); lab.set_color(INK)
    save(fig, "fig9_lexical_vs_semantic",
         "All hold-out questions, N = 1,500. The BERTScore axis does not start at 0; differences are small but significant (Section 4.13).")


def fig6():
    """Final model vs re-ranked copy across quartiles of asked-vs-copied question similarity (N = 300)."""
    q = SEM["by_copied_question_similarity"]
    keys = ["Q1", "Q2", "Q3", "Q4"]
    labels = ["Q1\n(%.2f)" % q[k]["asked_vs_copied_question_cos"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"wspace": 0.28})
    for ax, metric, title in ((axes[0], "rougeL", "ROUGE-L"), (axes[1], "bertscore_f1", "BERTScore F1")):
        for who, col, lab in (("final", FINAL, "Final model"), ("copy", COPY, "Copy (re-ranked)")):
            ys = [q[k]["%s|%s" % (metric, who)] for k in keys]
            ax.plot(range(4), ys, "-o", color=col, lw=2, ms=7, mec="white", mew=1.5, label=lab)
            ax.annotate(lab, (3, ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_xticks(range(4), [l.replace("Q1", k) for l, k in zip(labels, keys)])
        ax.set_xlim(-0.3, 4.1)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.supxlabel("Similarity of the copied training question to the asked question (quartile, mean cosine)", fontsize=8.5, color=INK2, y=-0.05)
    save(fig, "fig7_quartiles", "Development sample, N = 300 (about 75 questions per quartile). Notebook 10.", y=-0.12)


def fig7():
    steps, loss = zip(*TRAIN["eval_log"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(steps, loss, "-o", color=FINAL, lw=2, ms=6, mec="white", mew=1.5)
    for s, l in zip(steps, loss):
        if s in (steps[0], steps[-1]):
            ax.annotate("%.3f" % l, (s, l), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("Training step"); ax.set_ylabel("Dev loss")
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    save(fig, "fig6_dev_loss", "Adapter v2: {:,} training examples, 1 epoch ({} steps), 60 dev examples. Notebook 08.".format(TRAIN["n_train"], TRAIN["steps"]))


def fig8():
    R = SWEEP["results"]
    copy = R["B0a copy nearest answer"]["rougeL"]
    pts = []
    for k, v in R.items():
        if k.startswith("B0a"):
            continue
        name = "Phase 5" if k.startswith("Phase5") else k[0]
        pts.append((name, v["loops_8gram_x3"], v["rougeL"]))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.axhline(copy, color=INK2, lw=1, ls="--")
    ax.annotate("Copy nearest answer (%.4f)" % copy, (0.5, copy), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
    offs = {"Phase 5": (6, 4), "A": (6, -10), "B": (6, 0), "C": (6, -10), "D": (6, 2), "E": (-6, 7), "F": (6, 2), "G": (-6, -12)}
    for name, x, y in pts:
        chosen = name == "E"
        ax.plot(x, y, "o", ms=8, color=FINAL if chosen else NEUTRAL, mec="white", mew=1.5, zorder=3)
        dx, dy = offs.get(name, (6, 0))
        ax.annotate(name + (" (chosen)" if chosen else ""), (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha="right" if dx < 0 else "left", color=INK if chosen else INK2,
                    fontweight="bold" if chosen else "normal")
    ax.set_xlabel("Answers with repetition loops (of %d)" % SWEEP["n"]); ax.set_ylabel("ROUGE-L")
    ax.set_xlim(-0.5, 18)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    save(fig, "fig4_decoding", "N = %d hold-out questions, adapter v1, cap 1,280 tokens. Notebook 06." % SWEEP["n"])


def fig9():
    B = M11["bootstrap"]
    rows = [("final - copy re-ranked nearest answer", "Final - copy (re-ranked)"),
            ("final - copy nearest answer (original order)", "Final - copy (original order)"),
            ("final - Phase 4 model", "Final - fine-tuned v1 + retrieval"),
            ("final - no retrieval", "Final - fine-tuned, no retrieval"),
            ("final - zero-shot", "Final - zero-shot")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, metric, title in ((axes[0], "rougeL", "ROUGE-L difference"), (axes[1], "bertscore_f1", "BERTScore F1 difference")):
        for i, (k, _) in enumerate(rows):
            y = len(rows) - 1 - i
            d = B[metric][k]
            sig = d["lo"] > 0 or d["hi"] < 0
            ax.hlines(y, d["lo"], d["hi"], color=INK2, lw=1.5)
            ax.plot(d["diff"], y, "s", ms=7, color=FINAL if sig else "white", mec=FINAL, mew=1.5, zorder=3)
            left = d["diff"] < 0
            ax.annotate("%+.4f" % d["diff"], (d["lo"] if left else d["hi"], y), xytext=(-6 if left else 6, 0),
                        textcoords="offset points", va="center", ha="right" if left else "left", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_title(title)
        xgrid(ax)
        lo = min(B[metric][k]["lo"] for k, _ in rows); hi = max(B[metric][k]["hi"] for k, _ in rows)
        ax.set_xlim(min(lo, 0) - (hi - lo) * (0.22 if lo < 0 else 0.05), hi + (hi - lo) * 0.18)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), [lab for _, lab in rows][::-1])
    save(fig, "fig8_bootstrap", "Paired bootstrap, 2,000 resamples, N = 1,500. Squares: point estimate; lines: 95% CI; "
         "hollow = the interval includes 0 (not significant).")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig5(); fig6(); fig7(); fig8(); fig9()
