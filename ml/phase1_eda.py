# -*- coding: utf-8 -*-
import sys, io
# Force stdout to UTF-8 on Windows so Unicode chars print cleanly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
=============================================================================
  PHASE 1 — Exploratory Data Analysis (EDA)
  Chat Toxicity Detector | Jigsaw Toxic Comment Classification Dataset
=============================================================================
  Sections
  --------
  1. Load & Inspect
  2. Label Distribution Analysis
  3. Text-Length Exploration
  4. Visualisations (bar chart, histogram, co-occurrence heatmap)
  5. Summary / Findings
=============================================================================
"""

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from itertools import combinations
from collections import Counter

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "savefig.facecolor":"#0d1117",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#21262d",
    "font.family":      "DejaVu Sans",
})

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# colour palette (one per label, for consistent use across all plots)
LABEL_PALETTE = {
    "toxic":         "#ff6b6b",
    "severe_toxic":  "#ff9f43",
    "obscene":       "#feca57",
    "threat":        "#48dbfb",
    "insult":        "#ff9ff3",
    "identity_hate": "#54a0ff",
}
LABEL_COLORS = [LABEL_PALETTE[l] for l in LABEL_COLS]


def section(title: str) -> None:
    """Print a visually distinct section header."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def subsection(title: str) -> None:
    dashes = "-" * max(0, 65 - len(title))
    print(f"\n-- {title} {dashes}")


# =============================================================================
# 1. LOAD & INSPECT
# =============================================================================
section("1. LOAD & INSPECT")

train_path = os.path.join(DATA_DIR, "train.csv")
df = pd.read_csv(train_path)

subsection("Shape")
print(f"  Rows × Columns : {df.shape[0]:,} × {df.shape[1]}")

subsection("Column names & dtypes")
print(df.dtypes.to_string())

subsection("First 5 rows")
# Truncate long comment text for display
display_df = df.copy()
display_df["comment_text"] = display_df["comment_text"].str[:80] + "…"
print(display_df.head().to_string(index=False))

subsection("Null / missing values")
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(4)
null_report = pd.DataFrame({"null_count": null_counts, "null_%": null_pct})
print(null_report.to_string())
if null_counts.sum() == 0:
    print("\n  ✓ No missing values found in any column.")


# =============================================================================
# 2. LABEL DISTRIBUTION
# =============================================================================
section("2. LABEL DISTRIBUTION ANALYSIS")

subsection("Per-label counts & percentages")
label_stats = pd.DataFrame({
    "count":   df[LABEL_COLS].sum().astype(int),
    "percent": (df[LABEL_COLS].sum() / len(df) * 100).round(4),
}).sort_values("count", ascending=False)
print(label_stats.to_string())

subsection("Clean / non-toxic comments (zero labels)")
clean_mask  = (df[LABEL_COLS].sum(axis=1) == 0)
n_clean     = clean_mask.sum()
pct_clean   = n_clean / len(df) * 100
print(f"  Clean comments : {n_clean:,}  ({pct_clean:.2f}%)")
print(f"  Toxic comments : {len(df) - n_clean:,}  ({100 - pct_clean:.2f}%)")

subsection("Multi-label statistics")
label_sum = df[LABEL_COLS].sum(axis=1)
for k in range(1, len(LABEL_COLS) + 1):
    cnt = (label_sum == k).sum()
    if cnt > 0:
        print(f"  Exactly {k} label{'s' if k > 1 else ' '}: {cnt:,} comments")

subsection("Top 10 most common label combinations")
def combo_key(row):
    active = [col for col in LABEL_COLS if row[col] == 1]
    return "+".join(active) if active else "CLEAN"

df["label_combo"] = df[LABEL_COLS].apply(combo_key, axis=1)
combo_counts = df["label_combo"].value_counts().head(10)
print(combo_counts.to_string())


# =============================================================================
# 3. TEXT-LENGTH EXPLORATION
# =============================================================================
section("3. TEXT DATA EXPLORATION")

df["char_len"] = df["comment_text"].str.len()
df["word_len"] = df["comment_text"].str.split().str.len()

subsection("Character-length statistics")
char_stats = df["char_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
# rename for clarity
char_stats.index = ["count","mean","std","min","25%","median","75%","95%","99%","max"]
print(char_stats.apply(lambda x: f"{x:,.1f}").to_string())

subsection("Word-count statistics")
word_stats = df["word_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
word_stats.index = ["count","mean","std","min","25%","median","75%","95%","99%","max"]
print(word_stats.apply(lambda x: f"{x:,.1f}").to_string())

subsection("Extremely short comments (<= 5 chars)")
very_short = df[df["char_len"] <= 5]
print(f"  Count : {len(very_short):,}")
if len(very_short) > 0:
    print("  Examples:")
    for _, row in very_short.head(5).iterrows():
        print(f"    [{row['char_len']} chars] {repr(row['comment_text'])}")

subsection("Extremely long comments (>= 99th percentile)")
p99_char = int(df["char_len"].quantile(0.99))
very_long = df[df["char_len"] >= p99_char]
print(f"  99th-percentile threshold : {p99_char:,} chars")
print(f"  Count above threshold     : {len(very_long):,}")

subsection("Sample comments per label (2 examples each)")
for label in LABEL_COLS:
    subset = df[df[label] == 1]["comment_text"].dropna()
    print(f"\n  >> {label.upper()}")
    for i, text in enumerate(subset.head(2), 1):
        safe_text = text[:300].encode("ascii", errors="replace").decode("ascii")
        wrapped = textwrap.fill(safe_text, width=80, initial_indent="    ")
        print(f"  [{i}] {wrapped}")


# =============================================================================
# 4. VISUALISATIONS
# =============================================================================
section("4. VISUALISATIONS")
print("  Generating plots — saved to:", OUTPUT_DIR)

# ── 4a. Bar chart of label frequencies ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    label_stats.index,
    label_stats["count"],
    color=[LABEL_PALETTE[l] for l in label_stats.index],
    edgecolor="#30363d",
    linewidth=0.8,
    zorder=3,
)
ax.set_title("Label Frequencies in Training Set", fontsize=15, pad=14, color="#e6edf3", fontweight="bold")
ax.set_xlabel("Toxicity Label", fontsize=11)
ax.set_ylabel("Number of Comments", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", linewidth=0.6, zorder=0)
# annotate bars with count + %
for bar, (lbl, row) in zip(bars, label_stats.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{row['count']:,}\n({row['percent']:.1f}%)",
        ha="center", va="bottom", fontsize=8.5, color="#e6edf3"
    )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4a_label_frequency_bar.png"), dpi=150)
plt.close()
print("  ✓ Saved: 4a_label_frequency_bar.png")

# ── 4b. Histogram of comment lengths (chars) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Character length
ax = axes[0]
clipped_chars = df["char_len"].clip(upper=df["char_len"].quantile(0.99))
ax.hist(clipped_chars, bins=60, color="#48dbfb", edgecolor="#0d1117", linewidth=0.4, alpha=0.85)
ax.axvline(df["char_len"].median(), color="#feca57", linewidth=1.5, linestyle="--", label=f"Median = {df['char_len'].median():.0f}")
ax.axvline(df["char_len"].mean(),   color="#ff6b6b", linewidth=1.5, linestyle="--", label=f"Mean   = {df['char_len'].mean():.0f}")
ax.set_title("Character Length Distribution\n(clipped at 99th pct)", fontsize=11, color="#e6edf3", fontweight="bold")
ax.set_xlabel("Characters per comment")
ax.set_ylabel("Number of comments")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# Word length
ax = axes[1]
clipped_words = df["word_len"].clip(upper=df["word_len"].quantile(0.99))
ax.hist(clipped_words, bins=60, color="#ff9ff3", edgecolor="#0d1117", linewidth=0.4, alpha=0.85)
ax.axvline(df["word_len"].median(), color="#feca57", linewidth=1.5, linestyle="--", label=f"Median = {df['word_len'].median():.0f}")
ax.axvline(df["word_len"].mean(),   color="#ff6b6b", linewidth=1.5, linestyle="--", label=f"Mean   = {df['word_len'].mean():.0f}")
ax.set_title("Word Count Distribution\n(clipped at 99th pct)", fontsize=11, color="#e6edf3", fontweight="bold")
ax.set_xlabel("Words per comment")
ax.set_ylabel("Number of comments")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

fig.suptitle("Comment Length Distributions — Training Set", fontsize=13, color="#e6edf3", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4b_comment_length_histogram.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: 4b_comment_length_histogram.png")

# ── 4c. Label co-occurrence heatmap ──────────────────────────────────────────
cooccurrence = pd.DataFrame(
    np.zeros((len(LABEL_COLS), len(LABEL_COLS)), dtype=int),
    index=LABEL_COLS, columns=LABEL_COLS
)
for i, lbl_i in enumerate(LABEL_COLS):
    for j, lbl_j in enumerate(LABEL_COLS):
        cooccurrence.loc[lbl_i, lbl_j] = int((df[lbl_i] & df[lbl_j]).sum())

# Convert to percentages (relative to the row label's total)
cooccurrence_pct = cooccurrence.div(cooccurrence.values.diagonal(), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.eye(len(LABEL_COLS), dtype=bool)         # hide identical diagonal for readability
sns.heatmap(
    cooccurrence_pct,
    annot=True, fmt=".1f", annot_kws={"size": 10},
    cmap="YlOrRd",
    linewidths=0.5, linecolor="#0d1117",
    ax=ax,
    cbar_kws={"label": "% of row-label comments that also have col-label"},
    vmin=0, vmax=100,
)
ax.set_title(
    "Label Co-occurrence Heatmap\n(cell = % of row-label comments that ALSO carry col-label)",
    fontsize=11, color="#e6edf3", fontweight="bold", pad=12
)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4c_label_cooccurrence_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: 4c_label_cooccurrence_heatmap.png")

# ── 4d. Toxicity breakdown: stacked bar (clean vs each label) ────────────────
fig, ax = plt.subplots(figsize=(10, 5))
categories  = ["CLEAN"] + [l.replace("_", "\n") for l in LABEL_COLS]
counts      = [n_clean] + [int(df[l].sum()) for l in LABEL_COLS]
colors_all  = ["#2ea043"] + LABEL_COLORS
bars = ax.barh(categories, counts, color=colors_all, edgecolor="#0d1117", linewidth=0.6)
ax.set_title("Comment Category Breakdown", fontsize=13, color="#e6edf3", fontweight="bold")
ax.set_xlabel("Number of Comments")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", linestyle="--", linewidth=0.5, zorder=0)
for bar, cnt in zip(bars, counts):
    ax.text(
        bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
        f"{cnt:,}", va="center", fontsize=9, color="#e6edf3"
    )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4d_category_breakdown_hbar.png"), dpi=150)
plt.close()
print("  ✓ Saved: 4d_category_breakdown_hbar.png")


# =============================================================================
# 5. SUMMARY & FINDINGS
# =============================================================================
section("5. SUMMARY & FINDINGS")

rarest_label  = label_stats["count"].idxmin()
commonest_label = label_stats["count"].idxmax()
imbalance_ratio = label_stats["count"].max() / label_stats["count"].min()

print(f"""
  Dataset overview
  ────────────────
  • Total comments   : {len(df):,}
  • Label columns    : {', '.join(LABEL_COLS)}
  • Clean comments   : {n_clean:,}  ({pct_clean:.1f}% of total)
  • Toxic comments   : {len(df) - n_clean:,}  ({100 - pct_clean:.1f}% of total)

  Class balance
  ────────────────
  • The dataset is HEAVILY IMBALANCED.
    ~{pct_clean:.0f}% of comments are clean; toxic examples are a small minority.
  • Most common label : '{commonest_label}'  ({int(label_stats.loc[commonest_label,'count']):,} comments,  {label_stats.loc[commonest_label,'percent']:.2f}%)
  • Rarest label      : '{rarest_label}'   ({int(label_stats.loc[rarest_label,'count']):,} comments,  {label_stats.loc[rarest_label,'percent']:.2f}%)
  • Imbalance ratio (most/least frequent toxic label) : {imbalance_ratio:.1f}×

  Multi-label nature
  ────────────────
  • Many comments carry TWO or more labels simultaneously (e.g. 'toxic' + 'obscene').
  • This means the task is multi-label classification (not multi-class).
  • Challenges introduced:
      ① Standard accuracy is misleading — a model predicting all-zeros scores high.
      ② Each label needs its own probability threshold (not a single argmax).
      ③ Label correlation must be modelled (e.g. 'severe_toxic' almost always implies 'toxic').
      ④ Loss functions must handle partial/mixed labels (e.g. BinaryCrossEntropy per label).
      ⑤ Evaluation requires per-label metrics (AUC, F1) AND macro/micro averages.

  Real-world harm represented by each label
  ────────────────────────────────────────────
  • toxic         → General rudeness, disrespect, or hostility.
  • severe_toxic  → Extreme, aggressive hate speech or personal attacks.
  • obscene       → Sexually explicit or profane language.
  • threat        → Direct threats of violence or harm.
  • insult        → Demeaning language targeting a specific user.
  • identity_hate → Hate speech targeting race, religion, gender, sexual orientation, etc.

  Text-length insights
  ────────────────────
  • Median comment: {df['char_len'].median():.0f} chars / {df['word_len'].median():.0f} words
  • Long comments (≥ p99 = {p99_char:,} chars) : {len(very_long):,} — potential noise or spam.
  • Very short comments (≤ 5 chars)             : {len(very_short):,} — minimal context for the model.

  Recommended next steps (Phase 2)
  ────────────────────────────────
  • Text preprocessing: lowercase, remove HTML/URLs, strip special chars.
  • Handle class imbalance: oversampling (SMOTE-text), class weights, or focal loss.
  • Experiment with TF-IDF + Logistic Regression (strong baseline).
  • Progress to fine-tuning a transformer (DistilBERT) for best performance.
""")

print("  All plots saved to:", OUTPUT_DIR)
print("\n  EDA complete ✓\n")
