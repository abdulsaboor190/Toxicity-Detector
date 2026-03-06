# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
=============================================================================
  PHASE 5 - Deep Model Evaluation & Bias Auditing
  Chat Toxicity Detector | Jigsaw Toxic Comment Classification
=============================================================================
  JOB 1 : Deep model evaluation on original test set
  JOB 2 : Bias auditing with Jigsaw Unintended Bias dataset

Usage:
  python phase5_evaluation.py              # runs both JOB 1 & JOB 2
  python phase5_evaluation.py --job 1      # runs JOB 1 only
  python phase5_evaluation.py --job 2      # runs JOB 2 only
  python phase5_evaluation.py --bias-sample 50000  # sample bias dataset (CPU)
=============================================================================
"""
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--job", choices=["1", "2", "all"], default="all",
                     help="Which job(s) to run (default: all)")
_parser.add_argument("--bias-sample", type=int, default=None,
                     help="Limit bias dataset to N rows for faster CPU runs")
_args, _ = _parser.parse_known_args()
RUN_JOB1 = _args.job in ("1", "all")
RUN_JOB2 = _args.job in ("2", "all")

import os
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# The model used in Phase 4 (read from checkpoint filename)
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16  # inference only — no grad, lower memory

CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs/phase4/saved_models/bert_epoch1_f10.4322.pt")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "outputs/phase4/tuned_thresholds.json")
TEST_PATH       = os.path.join(BASE_DIR, "data/test.csv")
TEST_LABELS_PATH = os.path.join(BASE_DIR, "data/test_labels.csv")
BIAS_DATA_PATH  = os.path.join(BASE_DIR, "Second_Data/train.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/phase5/")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots/")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dark theme
MATPLOTLIB_DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "savefig.facecolor": "#0d1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "axes.edgecolor": "#30363d",
    "grid.color": "#21262d", "font.family": "DejaVu Sans",
    "axes.titlecolor": "#e6edf3",
}
matplotlib.rcParams.update(MATPLOTLIB_DARK)

COLORS = ["#ff6b6b", "#ff9f43", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff"]


# =============================================================================
# HELPERS
# =============================================================================
def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n-- {title} " + "-" * max(0, 65 - len(title)))

def ppath(rel):
    return os.path.join(BASE_DIR, rel)


# =============================================================================
# LOAD THRESHOLDS
# =============================================================================
section("SETUP")
with open(THRESHOLDS_PATH) as f:
    thresholds = json.load(f)
print(f"  Loaded thresholds: {thresholds}")


# =============================================================================
# INSPECT SECOND_DATA FOLDER
# =============================================================================
subsection("Second_Data contents")
for fname in sorted(os.listdir(os.path.join(BASE_DIR, "Second_Data"))):
    fpath = os.path.join(BASE_DIR, "Second_Data", fname)
    sz = os.path.getsize(fpath) / 1e6
    print(f"  {fname} -> {sz:.1f} MB")


# =============================================================================
# MODEL CLASS (self-contained — same as Phase 4)
# =============================================================================
class ToxicClassifier(nn.Module):
    """
    Transformer encoder + linear head for multi-label toxic comment classification.
    Compatible with BERT and DistilBERT.
    """
    def __init__(self, model_name, num_labels=6, dropout_rate=0.3):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls)
        return self.classifier(pooled)


class SimpleDataset(Dataset):
    """Minimal dataset for inference — no labels needed."""
    def __init__(self, texts, tokenizer, max_length):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc  = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# =============================================================================
# LOAD MODEL & TOKENIZER
# =============================================================================
subsection("Loading model & tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

print(f"  Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"  Loading model architecture: {MODEL_NAME}")
model = ToxicClassifier(MODEL_NAME, num_labels=len(LABEL_COLS))

if os.path.exists(CHECKPOINT_PATH):
    print(f"  Loading checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Checkpoint loaded (epoch={checkpoint.get('epoch', '?')}, "
              f"f1={checkpoint.get('val_f1_macro', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("  Checkpoint state dict loaded.")
else:
    print(f"  [WARN] Checkpoint not found at {CHECKPOINT_PATH}")
    print("  Using base DistilBERT weights (not fine-tuned).")

model = model.to(device)
model.eval()
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================
@torch.no_grad()
def run_inference(texts, batch_size=BATCH_SIZE, desc="Inference"):
    """Run model inference and return probability matrix [n, 6]."""
    dataset = SimpleDataset(texts, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs = []
    for batch in tqdm(loader, desc=f"  {desc}", unit="batch", ncols=90):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


# =============================================================================
# JOB 1: DEEP MODEL EVALUATION
# =============================================================================
if RUN_JOB1:
    section("JOB 1: DEEP MODEL EVALUATION")
    t_job1 = time.time()

    # ── Load test data ────────────────────────────────────────────────────
    subsection("Loading test data")
    df_test_raw = pd.read_csv(TEST_PATH)
    df_test_lbl = pd.read_csv(TEST_LABELS_PATH)
    df_test = df_test_raw.merge(df_test_lbl, on="id", how="inner")
    labeled_mask = (df_test[LABEL_COLS] != -1).any(axis=1)
    df_test = df_test[labeled_mask].reset_index(drop=True)
    print(f"  Test set (labeled): {len(df_test):,} rows")

    y_test = df_test[LABEL_COLS].values.astype(int)
    test_texts = df_test["comment_text"].fillna("").tolist()

    # ── 1A: Generate Test Predictions ─────────────────────────────────────
    subsection("1A: Generate Test Predictions")
    test_probs = run_inference(test_texts, desc="Test inference")
    print(f"  Probabilities shape: {test_probs.shape}")

    # Apply thresholds
    test_preds = np.zeros_like(test_probs, dtype=int)
    for i, lbl in enumerate(LABEL_COLS):
        test_preds[:, i] = (test_probs[:, i] >= thresholds[lbl]).astype(int)

    np.save(os.path.join(OUTPUT_DIR, "test_probabilities.npy"), test_probs)
    np.save(os.path.join(OUTPUT_DIR, "test_predictions.npy"), test_preds)
    print("  Saved: test_probabilities.npy, test_predictions.npy")

    # ── 1B: ROC-AUC Per Label ─────────────────────────────────────────────
    subsection("1B: ROC-AUC Per Label")
    fig, ax = plt.subplots(figsize=(10, 8))
    auc_scores = {}

    print(f"\n  {'Label':<16} {'AUC':>8}  {'Operating TPR':>14}  {'Operating FPR':>14}")
    print(f"  {'-'*56}")

    for i, lbl in enumerate(LABEL_COLS):
        yt = y_test[:, i]
        yp = test_probs[:, i]

        if yt.sum() == 0 or yt.sum() == len(yt):
            print(f"  {lbl:<16}  N/A (no positive/negative samples)")
            auc_scores[lbl] = 0.0
            continue

        auc = roc_auc_score(yt, yp)
        auc_scores[lbl] = auc
        fpr, tpr, roc_thresh = roc_curve(yt, yp)

        ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
                label=f"{lbl} (AUC = {auc:.3f})")

        # Operating point
        t = thresholds[lbl]
        pred_at_t = (yp >= t).astype(int)
        op_tpr = recall_score(yt, pred_at_t, zero_division=0)
        op_fpr = ((pred_at_t == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)
        ax.scatter([op_fpr], [op_tpr], color=COLORS[i], s=80, zorder=5,
                   edgecolor="white", linewidth=1.5)

        print(f"  {lbl:<16} {auc:8.4f}  {op_tpr:14.4f}  {op_fpr:14.4f}")

    macro_auc = np.mean(list(auc_scores.values()))
    print(f"  {'MACRO AVG':<16} {macro_auc:8.4f}")

    ax.plot([0, 1], [0, 1], color="#8b949e", linewidth=1, linestyle="--",
            label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves -- Per Label (BERT fine-tuned)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/roc_curves.png")

    # ── 1C: Precision-Recall Curves ───────────────────────────────────────
    subsection("1C: Precision-Recall Curves")
    # KEY INSIGHT:
    # For severely imbalanced labels (threat ~0.3%, identity_hate ~1.5%),
    # PR curves are MORE informative than ROC curves because:
    # 1. ROC-AUC can be misleadingly high when negatives dominate (the
    #    FPR denominator is huge, so even many FPs barely move FPR).
    # 2. PR curves focus on the positive class — if precision drops at
    #    useful recall levels, the model is not actually useful despite
    #    a seemingly high ROC-AUC.

    fig, ax = plt.subplots(figsize=(10, 8))
    ap_scores = {}

    for i, lbl in enumerate(LABEL_COLS):
        yt = y_test[:, i]
        yp = test_probs[:, i]
        if yt.sum() == 0:
            ap_scores[lbl] = 0.0
            continue

        prec, rec, pr_thresh = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ap_scores[lbl] = ap

        ax.plot(rec, prec, color=COLORS[i], linewidth=2,
                label=f"{lbl} (AP = {ap:.3f})")

        # Random baseline = positive rate
        base_rate = yt.mean()
        ax.axhline(base_rate, color=COLORS[i], linewidth=0.8, linestyle=":",
                   alpha=0.5)

        # Operating point
        t = thresholds[lbl]
        pred_at_t = (yp >= t).astype(int)
        op_prec = precision_score(yt, pred_at_t, zero_division=0)
        op_rec  = recall_score(yt, pred_at_t, zero_division=0)
        ax.scatter([op_rec], [op_prec], color=COLORS[i], s=80, zorder=5,
                   edgecolor="white", linewidth=1.5)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves -- Per Label", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/pr_curves.png")

    # ── 1D: Confusion Matrix Per Label ────────────────────────────────────
    subsection("1D: Confusion Matrices")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    diag_rows = []
    print(f"\n  {'Label':<16} {'TP':>5} {'FP':>5} {'TN':>6} {'FN':>5} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'MissRate':>9}")
    print(f"  {'-'*75}")

    for i, lbl in enumerate(LABEL_COLS):
        yt = y_test[:, i]
        yp = test_preds[:, i]

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        miss_rate = fn / max(tp + fn, 1)

        diag_rows.append({
            "label": lbl, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1, "miss_rate": miss_rate
        })

        print(f"  {lbl:<16} {tp:5d} {fp:5d} {tn:6d} {fn:5d} "
              f"{prec:7.4f} {rec:7.4f} {f1:7.4f} {miss_rate:9.4f}")

        # Plot confusion matrix
        ax = axes[i]
        # Row-normalized
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm_norm / row_sums * 100

        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="YlOrRd", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    annot_kws={"fontsize": 12, "fontweight": "bold"})

        # Add percentage annotations
        for r in range(2):
            for c in range(2):
                ax.text(c + 0.5, r + 0.75, f"({cm_pct[r, c]:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="#e6edf3", alpha=0.8)

        ax.set_title(f"{lbl.replace('_', ' ').title()} (F1 = {f1:.3f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices -- Per Label (BERT fine-tuned)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: plots/confusion_matrices.png")

    # ── 1E: Error Analysis ────────────────────────────────────────────────
    subsection("1E: Error Analysis")

    # False Positives for 'toxic': model says toxic, truth says clean
    toxic_idx = LABEL_COLS.index("toxic")
    fp_mask = (test_preds[:, toxic_idx] == 1) & (y_test[:, toxic_idx] == 0)
    fp_probs = test_probs[fp_mask, toxic_idx]
    fp_indices = np.where(fp_mask)[0]
    fp_sort = fp_probs.argsort()[::-1][:20]
    fp_top = fp_indices[fp_sort]

    print(f"\n  Top 20 FALSE POSITIVES (toxic label):")
    print(f"  Model flagged as toxic, but ground truth = 0")
    error_rows = []
    for rank, idx in enumerate(fp_top, 1):
        text = str(df_test.iloc[idx]["comment_text"])[:120]
        prob = test_probs[idx, toxic_idx]
        true_labels = ", ".join([l for l in LABEL_COLS if y_test[idx, LABEL_COLS.index(l)] == 1])
        if not true_labels:
            true_labels = "clean"
        print(f"  [{rank:2d}] prob={prob:.4f} | true={true_labels}")
        print(f"       {text}...")
        error_rows.append({
            "comment_text": df_test.iloc[idx]["comment_text"],
            "true_labels": true_labels,
            "pred_proba": prob,
            "error_type": "false_positive_toxic",
        })

    # False Negatives for 'threat': model misses actual threats
    threat_idx = LABEL_COLS.index("threat")
    fn_mask = (test_preds[:, threat_idx] == 0) & (y_test[:, threat_idx] == 1)
    fn_probs = test_probs[fn_mask, threat_idx]
    fn_indices = np.where(fn_mask)[0]
    fn_sort = fn_probs.argsort()[:20]  # lowest prob first = worst misses
    if len(fn_sort) > 0:
        fn_top = fn_indices[fn_sort]
    else:
        fn_top = []

    print(f"\n  Top 20 FALSE NEGATIVES (threat label):")
    print(f"  Actual threats the model missed")
    for rank, idx in enumerate(fn_top, 1):
        text = str(df_test.iloc[idx]["comment_text"])[:120]
        prob = test_probs[idx, threat_idx]
        true_labels = ", ".join([l for l in LABEL_COLS if y_test[idx, LABEL_COLS.index(l)] == 1])
        print(f"  [{rank:2d}] prob={prob:.4f} | true={true_labels}")
        print(f"       {text}...")
        error_rows.append({
            "comment_text": df_test.iloc[idx]["comment_text"],
            "true_labels": true_labels,
            "pred_proba": prob,
            "error_type": "false_negative_threat",
        })

    # Save error analysis
    df_errors = pd.DataFrame(error_rows)
    error_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
    df_errors.to_csv(error_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: error_analysis.csv ({len(df_errors)} rows)")

    # Compute per-label F1 for summary
    f1_per_label = {}
    miss_per_label = {}
    for i, lbl in enumerate(LABEL_COLS):
        f1_per_label[lbl] = f1_score(y_test[:, i], test_preds[:, i], zero_division=0)
        fn_count = ((test_preds[:, i] == 0) & (y_test[:, i] == 1)).sum()
        tp_count = ((test_preds[:, i] == 1) & (y_test[:, i] == 1)).sum()
        miss_per_label[lbl] = fn_count / max(fn_count + tp_count, 1)

    macro_f1 = np.mean(list(f1_per_label.values()))

    t_job1_end = time.time()
    print(f"\n  JOB 1 completed in {(t_job1_end - t_job1)/60:.1f} minutes")

else:
    section("JOB 1: SKIPPED (--job 2 was set)")
    auc_scores = {}
    f1_per_label = {}
    miss_per_label = {}
    macro_auc = 0.0
    macro_f1 = 0.0


# =============================================================================
# JOB 2: BIAS AUDITING
# =============================================================================
if RUN_JOB2:
    section("JOB 2: BIAS AUDITING")
    t_job2 = time.time()

    # IDENTITY PROXY BIAS:
    # Toxicity models trained on human-annotated data often learn spurious correlations.
    # If training data contains many toxic comments about Group X, the model may learn
    # that mentions of Group X -> toxic, even for neutral comments like "I am Muslim."
    # This is called identity proxy bias. It causes real harm: legitimate speech by or
    # about marginalized groups gets systematically over-flagged.
    # We measure this using Subgroup AUC from the Jigsaw Bias evaluation framework.

    # ── 2B: Prepare the Bias Dataset ──────────────────────────────────────
    subsection("2B: Load & prepare bias dataset")
    t0 = time.time()
    bias_cols_to_load = None  # Load all columns first time to discover identity cols

    print(f"  Loading {BIAS_DATA_PATH} ...")
    df_bias = pd.read_csv(BIAS_DATA_PATH, engine="c", low_memory=False)
    print(f"  Shape: {df_bias.shape}")
    print(f"  Columns: {list(df_bias.columns)}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Binarize target
    df_bias["y_toxic"] = (df_bias["target"] >= 0.5).astype(int)
    print(f"\n  Target distribution:")
    print(f"    toxic (target >= 0.5): {df_bias['y_toxic'].sum():,} "
          f"({df_bias['y_toxic'].mean()*100:.1f}%)")
    print(f"    clean (target < 0.5):  {(1 - df_bias['y_toxic']).sum():,} "
          f"({(1 - df_bias['y_toxic'].mean())*100:.1f}%)")

    # Identify ALL identity columns (float columns between 0-1, exclude common non-identity ones)
    EXCLUDE_COLS = {
        "id", "target", "comment_text", "y_toxic", "pred_toxic_proba",
        "created_date", "publication_id", "parent_id", "article_id",
        "rating", "funny", "wow", "sad", "likes", "disagree",
        "sexual_explicit", "obscene", "threat", "insult", "identity_attack",
        "severe_toxicity", "toxicity_annotator_count", "identity_annotator_count",
    }
    IDENTITY_COLS = []
    for col in df_bias.columns:
        if col in EXCLUDE_COLS:
            continue
        if df_bias[col].dtype in ["float64", "float32"]:
            vals = df_bias[col].dropna()
            if len(vals) > 0 and vals.min() >= 0 and vals.max() <= 1.0:
                IDENTITY_COLS.append(col)

    print(f"\n  Discovered {len(IDENTITY_COLS)} identity columns:")
    for col in IDENTITY_COLS:
        n_mentioned = (df_bias[col].fillna(0) >= 0.5).sum()
        if n_mentioned > 100:
            print(f"    {col:<40} {n_mentioned:>8,} mentions")

    # Binarize identity columns
    for col in IDENTITY_COLS:
        df_bias[f"id_{col}"] = (df_bias[col].fillna(0) >= 0.5).astype(int)

    # Filter to identity columns with > 100 mentions
    IDENTITY_COLS_FILTERED = [
        col for col in IDENTITY_COLS
        if df_bias[f"id_{col}"].sum() > 100
    ]
    print(f"\n  Identity groups with > 100 mentions: {len(IDENTITY_COLS_FILTERED)}")

    # Print first 5 rows
    print(f"\n  First 5 rows (key columns):")
    display_cols = ["comment_text", "target", "y_toxic"] + IDENTITY_COLS_FILTERED[:5]
    existing_display = [c for c in display_cols if c in df_bias.columns]
    print(df_bias[existing_display].head().to_string(max_colwidth=60))

    # ── Optional: sample bias dataset for CPU ─────────────────────────────
    if _args.bias_sample is not None and _args.bias_sample < len(df_bias):
        df_bias = df_bias.sample(n=_args.bias_sample, random_state=42).reset_index(drop=True)
        print(f"\n  [bias-sample] Sampled {len(df_bias):,} rows for faster processing.")

    # ── Run BERT inference on bias dataset ────────────────────────────────
    subsection("2B: BERT inference on bias dataset")
    bias_texts = df_bias["comment_text"].fillna("").tolist()
    print(f"  Running inference on {len(bias_texts):,} comments ...")

    bias_probs = run_inference(bias_texts, batch_size=BATCH_SIZE, desc="Bias inference")
    # We only need the 'toxic' label probability (index 0)
    df_bias["pred_toxic_proba"] = bias_probs[:, 0]

    print(f"  Inference complete. Mean toxic proba: {df_bias['pred_toxic_proba'].mean():.4f}")

    # ── 2C: Subgroup AUC Computation ──────────────────────────────────────
    subsection("2C: Subgroup AUC Computation")

    def compute_subgroup_metrics(df, identity_col, proba_col="pred_toxic_proba",
                                  label_col="y_toxic"):
        """
        Compute Jigsaw Bias framework metrics for one identity group.

        Returns:
            dict with subgroup_auc, bpsn_auc, bnsp_auc, subgroup_size
        """
        id_col = f"id_{identity_col}"
        result = {"identity": identity_col, "subgroup_size": int(df[id_col].sum())}

        # 1. SUBGROUP AUC: how well does model rank within this group?
        sub = df[df[id_col] == 1]
        if sub[label_col].nunique() < 2 or len(sub) < 10:
            result["subgroup_auc"] = np.nan
        else:
            result["subgroup_auc"] = roc_auc_score(sub[label_col], sub[proba_col])

        # 2. BPSN AUC: (identity=1 & toxic=0) UNION (identity=0 & toxic=1)
        bpsn = pd.concat([
            df[(df[id_col] == 1) & (df[label_col] == 0)],
            df[(df[id_col] == 0) & (df[label_col] == 1)],
        ])
        if bpsn[label_col].nunique() < 2 or len(bpsn) < 10:
            result["bpsn_auc"] = np.nan
        else:
            result["bpsn_auc"] = roc_auc_score(bpsn[label_col], bpsn[proba_col])

        # 3. BNSP AUC: (identity=1 & toxic=1) UNION (identity=0 & toxic=0)
        bnsp = pd.concat([
            df[(df[id_col] == 1) & (df[label_col] == 1)],
            df[(df[id_col] == 0) & (df[label_col] == 0)],
        ])
        if bnsp[label_col].nunique() < 2 or len(bnsp) < 10:
            result["bnsp_auc"] = np.nan
        else:
            result["bnsp_auc"] = roc_auc_score(bnsp[label_col], bnsp[proba_col])

        return result

    # Overall AUC
    overall_auc = roc_auc_score(df_bias["y_toxic"], df_bias["pred_toxic_proba"])
    print(f"  Overall AUC: {overall_auc:.4f}")

    # Compute for ALL identity groups
    bias_results = []
    for col in tqdm(IDENTITY_COLS_FILTERED, desc="  Subgroup metrics", ncols=90):
        metrics = compute_subgroup_metrics(df_bias, col)
        bias_results.append(metrics)

    df_bias_results = pd.DataFrame(bias_results)
    df_bias_results = df_bias_results.sort_values("subgroup_auc", ascending=True)

    # Print results table
    print(f"\n  {'Identity':<42} {'Sub AUC':>8} {'BPSN':>8} {'BNSP':>8} {'Size':>8}")
    print(f"  {'-'*78}")
    for _, row in df_bias_results.iterrows():
        sg = f"{row['subgroup_auc']:.4f}" if pd.notna(row['subgroup_auc']) else "N/A"
        bp = f"{row['bpsn_auc']:.4f}" if pd.notna(row['bpsn_auc']) else "N/A"
        bn = f"{row['bnsp_auc']:.4f}" if pd.notna(row['bnsp_auc']) else "N/A"
        print(f"  {row['identity']:<42} {sg:>8} {bp:>8} {bn:>8} {row['subgroup_size']:>8,}")

    # Save
    bias_csv_path = os.path.join(OUTPUT_DIR, "bias_audit_results.csv")
    df_bias_results.to_csv(bias_csv_path, index=False)
    print(f"\n  Saved: bias_audit_results.csv")

    # ── 2D: Bias Audit Visualization ──────────────────────────────────────
    subsection("2D: Bias Audit Visualization")

    # PLOT 1 — Subgroup AUC Bar Chart
    df_plot = df_bias_results.dropna(subset=["subgroup_auc"]).copy()
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.4)))

    bar_colors = []
    for _, row in df_plot.iterrows():
        sg = row["subgroup_auc"]
        if sg < overall_auc - 0.05:
            bar_colors.append("#ff6b6b")  # Red — significant bias
        elif sg < overall_auc:
            bar_colors.append("#feca57")  # Yellow — mild
        else:
            bar_colors.append("#54a0ff")  # Green/blue — good

    ax.barh(range(len(df_plot)), df_plot["subgroup_auc"].values,
            color=bar_colors, edgecolor="#0d1117", linewidth=0.5)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["identity"].values, fontsize=9)
    ax.axvline(overall_auc, color="#ff6b6b", linewidth=2, linestyle="--",
               label=f"Overall AUC = {overall_auc:.3f}")

    for idx, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(row["subgroup_auc"] + 0.002, idx,
                f"n={row['subgroup_size']:,}", va="center", fontsize=7,
                color="#8b949e")

    ax.set_xlabel("Subgroup AUC", fontsize=12)
    ax.set_title("Subgroup AUC by Identity Group -- Red = Significant Bias Detected",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "subgroup_auc.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/subgroup_auc.png")

    # PLOT 2 — Bias Heatmap
    heatmap_cols = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    df_hm = df_plot.set_index("identity")[heatmap_cols].astype(float)
    df_hm.columns = ["Subgroup AUC", "BPSN AUC", "BNSP AUC"]

    fig, ax = plt.subplots(figsize=(8, max(6, len(df_hm) * 0.35)))
    sns.heatmap(df_hm, annot=True, fmt=".3f", ax=ax,
                cmap="RdYlGn", center=overall_auc,
                linewidths=0.5, linecolor="#30363d",
                cbar_kws={"label": "AUC", "shrink": 0.8})
    ax.set_title("Bias Audit Heatmap -- Jigsaw Unintended Bias Framework",
                 fontsize=13, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bias_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/bias_heatmap.png")

    # PLOT 3 — FPR by Identity
    toxic_thresh = thresholds["toxic"]
    fpr_data = []
    for col in IDENTITY_COLS_FILTERED:
        id_col = f"id_{col}"
        sub = df_bias[df_bias[id_col] == 1]
        sub_pred = (sub["pred_toxic_proba"] >= toxic_thresh).astype(int)
        sub_true = sub["y_toxic"]
        fp = ((sub_pred == 1) & (sub_true == 0)).sum()
        tn = ((sub_pred == 0) & (sub_true == 0)).sum()
        fpr_val = fp / max(fp + tn, 1)
        fpr_data.append({"identity": col, "fpr": fpr_val, "n": int(sub.shape[0])})

    df_fpr = pd.DataFrame(fpr_data).sort_values("fpr", ascending=False)

    # Overall FPR
    overall_pred = (df_bias["pred_toxic_proba"] >= toxic_thresh).astype(int)
    overall_fp = ((overall_pred == 1) & (df_bias["y_toxic"] == 0)).sum()
    overall_tn = ((overall_pred == 0) & (df_bias["y_toxic"] == 0)).sum()
    overall_fpr = overall_fp / max(overall_fp + overall_tn, 1)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_fpr) * 0.4)))
    bar_colors_fpr = ["#ff6b6b" if r["fpr"] > overall_fpr * 1.5 else
                      "#feca57" if r["fpr"] > overall_fpr else
                      "#54a0ff" for _, r in df_fpr.iterrows()]
    ax.barh(range(len(df_fpr)), df_fpr["fpr"].values,
            color=bar_colors_fpr, edgecolor="#0d1117", linewidth=0.5)
    ax.set_yticks(range(len(df_fpr)))
    ax.set_yticklabels(df_fpr["identity"].values, fontsize=9)
    ax.axvline(overall_fpr, color="#ff6b6b", linewidth=2, linestyle="--",
               label=f"Overall FPR = {overall_fpr:.4f}")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_title("False Positive Rate by Identity Group", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fpr_by_identity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/fpr_by_identity.png")

    # ── 2E: Bias Audit Summary Report ─────────────────────────────────────
    subsection("2E: Bias Audit Summary Report")

    # Find best/worst
    valid_rows = df_bias_results.dropna(subset=["subgroup_auc"])
    worst_row = valid_rows.iloc[0]   # sorted ascending
    best_row  = valid_rows.iloc[-1]

    sig_bias_threshold = overall_auc - 0.05
    groups_with_bias = valid_rows[valid_rows["subgroup_auc"] < sig_bias_threshold]["identity"].tolist()

    # Most over-flagged (highest FPR)
    most_overflagged = df_fpr.iloc[0]
    # Least protected (highest miss rate for identity mentioned)
    # We can approximate by looking at groups with low BNSP
    bnsp_valid = df_bias_results.dropna(subset=["bnsp_auc"]).sort_values("bnsp_auc")
    least_protected = bnsp_valid.iloc[0] if len(bnsp_valid) > 0 else None

    # Determine dominant bias type
    mean_bpsn = df_bias_results["bpsn_auc"].mean()
    mean_bnsp = df_bias_results["bnsp_auc"].mean()
    if mean_bpsn < mean_bnsp:
        bias_dominance = "BPSN (over-flagging neutral mentions) is the dominant bias type"
    else:
        bias_dominance = "BNSP (under-flagging hate targeting groups) is the dominant bias type"

    report_date = datetime.now().strftime("%Y-%m-%d")
    mean_sub_auc = valid_rows["subgroup_auc"].mean()

    report = f"""## BIAS AUDIT REPORT -- Chat Toxicity Detector
### Model: {MODEL_NAME} fine-tuned on Jigsaw Toxic Comments
### Date: {report_date}

---

#### Overall Performance
- Overall AUC: {overall_auc:.4f}
- Mean Subgroup AUC: {mean_sub_auc:.4f}

#### Subgroup AUC Results
- Best performing group: {best_row['identity']} (AUC = {best_row['subgroup_auc']:.4f})
- Worst performing group: {worst_row['identity']} (AUC = {worst_row['subgroup_auc']:.4f})
- Groups with significant bias (Subgroup AUC < {sig_bias_threshold:.3f}): {', '.join(groups_with_bias) if groups_with_bias else 'None'}

#### Key Findings
1. **Over-flagging**: The most over-flagged identity group is `{most_overflagged['identity']}` (FPR = {most_overflagged['fpr']:.4f} vs overall FPR = {overall_fpr:.4f}).
2. **Under-flagging**: {'The least protected group is `' + least_protected['identity'] + '` (BNSP AUC = ' + f"{least_protected['bnsp_auc']:.4f}" + ').' if least_protected is not None else 'N/A'}
3. **Dominant bias type**: {bias_dominance}. Mean BPSN AUC = {mean_bpsn:.4f}, Mean BNSP AUC = {mean_bnsp:.4f}.

#### Known Limitations
- This model was trained on Wikipedia comments and may not generalize to other domains (social media, gaming chat, etc.)
- Annotation reflects the perspectives of the annotators (predominantly English-speaking, US-based)
- Subgroup metrics are only computed for groups with > 100 mentions in the bias dataset
- Identity columns are based on annotator agreement (threshold >= 0.5) and may miss subtle mentions
- The bias dataset itself may contain annotation biases

#### Recommendations
- Apply threshold calibration per identity subgroup to equalize FPR across groups
- Consider debiasing techniques such as counterfactual data augmentation
- Regularly re-evaluate on updated bias datasets as language patterns evolve
- For deployment: implement human-in-the-loop review for comments mentioning identity terms
- {"Given significant bias detected in " + str(len(groups_with_bias)) + " groups, additional debiasing is recommended before production deployment." if groups_with_bias else "No significant subgroup bias detected, but continued monitoring is recommended."}
"""
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "bias_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: bias_report.md")

    t_job2_end = time.time()
    print(f"\n  JOB 2 completed in {(t_job2_end - t_job2)/60:.1f} minutes")

else:
    section("JOB 2: SKIPPED (--job 1 was set)")
    overall_auc = 0.0
    mean_sub_auc = 0.0
    groups_with_bias = []
    worst_row = {"identity": "N/A", "subgroup_auc": 0.0}
    best_row  = {"identity": "N/A", "subgroup_auc": 0.0}


# =============================================================================
# FINAL PHASE 5 SUMMARY
# =============================================================================
section("PHASE 5 SUMMARY")

summary = {
    "evaluation": {
        "roc_auc_per_label": {k: float(v) for k, v in auc_scores.items()} if auc_scores else {},
        "macro_auc": float(macro_auc),
        "f1_per_label": {k: float(v) for k, v in f1_per_label.items()} if f1_per_label else {},
        "macro_f1": float(macro_f1),
        "miss_rate_per_label": {k: float(v) for k, v in miss_per_label.items()} if miss_per_label else {},
    },
    "bias_audit": {
        "overall_auc": float(overall_auc),
        "worst_subgroup": {
            "name": str(worst_row.get("identity", worst_row["identity"]) if isinstance(worst_row, dict) else worst_row["identity"]),
            "subgroup_auc": float(worst_row.get("subgroup_auc", worst_row["subgroup_auc"]) if isinstance(worst_row, dict) else worst_row["subgroup_auc"]),
        },
        "best_subgroup": {
            "name": str(best_row.get("identity", best_row["identity"]) if isinstance(best_row, dict) else best_row["identity"]),
            "subgroup_auc": float(best_row.get("subgroup_auc", best_row["subgroup_auc"]) if isinstance(best_row, dict) else best_row["subgroup_auc"]),
        },
        "groups_with_significant_bias": groups_with_bias if isinstance(groups_with_bias, list) else [],
        "mean_subgroup_auc": float(mean_sub_auc) if isinstance(mean_sub_auc, (int, float)) else 0.0,
    },
    "config": {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "thresholds": thresholds,
    },
}

summary_path = os.path.join(OUTPUT_DIR, "phase5_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"  Summary saved: {summary_path}")
print(f"\n  Phase 5 complete!")
