# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
=============================================================================
  PHASE 4 - Model Training & Evaluation
  Chat Toxicity Detector | Jigsaw Toxic Comment Classification
=============================================================================
  STAGE 1 : TF-IDF + Logistic Regression + LightGBM (Baseline)
  STAGE 2 : BERT fine-tuning (bert-base-uncased)

Usage:
  python phase4_models.py              # runs Stage 1 only (fast, CPU-friendly)
  python phase4_models.py --stage 2    # runs Stage 2 only (needs checkpoint)
  python phase4_models.py --stage all  # runs both stages end-to-end
=============================================================================
"""
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--stage", choices=["1", "2", "all"], default="1",
                     help="Which stage(s) to run (default: 1)")
_parser.add_argument("--wandb", choices=["disabled", "online", "offline"],
                     default="disabled",
                     help="wandb mode (default: disabled — no login required)")
_parser.add_argument("--bert-epochs", type=int, default=None,
                     help="Override BERT epochs (default: 3)")
_parser.add_argument("--bert-sample", type=int, default=None,
                     help="Limit BERT training to N rows (e.g. 5000 for CPU quick-run). "
                          "If not set, uses all training data.")
_parser.add_argument("--bert-model", type=str, default=None,
                     help="Override BERT model name (e.g. distilbert-base-uncased)")
_args, _ = _parser.parse_known_args()
RUN_STAGE1 = _args.stage in ("1", "all")
RUN_STAGE2 = _args.stage in ("2", "all")
WANDB_MODE = _args.wandb

import os
import sys
import json
import time
import pickle
import random
import warnings
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Seeds (full reproducibility) ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Import torch (with graceful error) ───────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import (
        AutoModel, AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    TORCH_OK = True
except ImportError as e:
    print(f"  [WARN] PyTorch/Transformers not available: {e}")
    print("  Stage 2 (BERT) will be skipped.")
    TORCH_OK = False

# ── Import LightGBM ───────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    print("  [WARN] LightGBM not found. Stage 1C will be skipped.")
    LGBM_OK = False

# ── Import wandb (optional) ───────────────────────────────────────────────────
try:
    import wandb
    WANDB_OK = True
    if WANDB_MODE == "disabled":
        # Fully offline — no login prompt, no network calls
        import os as _os
        _os.environ["WANDB_MODE"] = "disabled"
except ImportError:
    WANDB_OK = False
    print("  [WARN] wandb not found — metrics will only be printed locally.")

# ── Import iterstrat for multi-label stratified split ────────────────────────
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    ITERSTRAT_OK = True
except ImportError:
    ITERSTRAT_OK = False

# =============================================================================
# CENTRAL CONFIGURATION
# =============================================================================
config = {
    # Data
    "train_path":       "outputs/phase3/train_augmented.csv",
    "test_path":        "data/test.csv",
    "test_labels_path": "data/test_labels.csv",
    "label_cols":       ["toxic", "severe_toxic", "obscene",
                         "threat", "insult", "identity_hate"],
    # Split
    "val_size":    0.15,
    "random_seed": 42,
    # TF-IDF
    "tfidf_max_features": 50000,
    "tfidf_ngram_range":  (1, 2),
    # Transformer
    # NOTE: distilbert-base-uncased is used by default — it is 40% smaller and
    # 60% faster than bert-base-uncased with ~97% of the accuracy on most tasks.
    # Switch to bert-base-uncased with --bert-model bert-base-uncased if needed.
    "model_name":          "distilbert-base-uncased",
    "max_length":          128,   # 128 is safe on CPU; GPU can handle 256
    "batch_size":          8,     # micro-batch for CPU; GPU uses 32
    "grad_accum_steps":    2,     # effective batch = batch_size * grad_accum_steps
    "epochs":              3,     # use --bert-epochs N to override
    "learning_rate":       2e-5,
    "warmup_ratio":        0.1,
    "gradient_clip":       1.0,
    "cpu_subset_frac":     1.0,  # set <1.0 to train BERT on a fraction of data on CPU
    "bert_sample":         None,  # set via --bert-sample N
    # Output
    "output_dir":     "outputs/phase4/",
    "model_save_dir": "outputs/phase4/saved_models/",
    "plots_dir":      "outputs/phase4/plots/",
}

# Apply CLI overrides
if _args.bert_epochs is not None:
    config["epochs"] = _args.bert_epochs
    print(f"  [CLI] BERT epochs overridden to {config['epochs']}.")
if _args.bert_model is not None:
    config["model_name"] = _args.bert_model
    print(f"  [CLI] BERT model overridden to: {config['model_name']}")
if _args.bert_sample is not None:
    config["bert_sample"] = _args.bert_sample
    print(f"  [CLI] BERT training sample capped at {config['bert_sample']:,} rows.")

LABEL_COLS = config["label_cols"]
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

for d in [config["output_dir"], config["model_save_dir"], config["plots_dir"]]:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

MATPLOTLIB_DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "savefig.facecolor": "#0d1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "axes.edgecolor": "#30363d",
    "grid.color": "#21262d", "font.family": "DejaVu Sans",
    "axes.titlecolor": "#e6edf3",
}
matplotlib.rcParams.update(MATPLOTLIB_DARK)


# =============================================================================
# HELPERS
# =============================================================================
def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n-- {title} " + "-" * max(0, 65 - len(title)))

def ppath(rel):
    """Return absolute path from a relative path."""
    return os.path.join(BASE_DIR, rel)

def safe_log(metrics: dict):
    """Log to wandb if available, always print locally."""
    print("  Metrics:", {k: f"{v:.4f}" if isinstance(v, float) else v
                         for k, v in metrics.items()})
    if WANDB_OK:
        wandb.log(metrics)


# =============================================================================
# LOAD CLASS WEIGHTS
# =============================================================================
weights_path = ppath("outputs/phase3/class_weights.json")
with open(weights_path) as f:
    class_weights_dict = json.load(f)
print(f"  Loaded class weights: {class_weights_dict}")

if TORCH_OK:
    pos_weight_tensor = torch.FloatTensor(
        [class_weights_dict[l] for l in LABEL_COLS]
    )
    print(f"  pos_weight tensor: {pos_weight_tensor.tolist()}")


# =============================================================================
# LOAD tune_thresholds FROM PHASE 3
# =============================================================================
def tune_thresholds(y_true, y_pred_proba, label_cols, save_path=None):
    """
    Search [0.10, 0.90] in steps of 0.01 for the threshold maximising F1
    on each label independently.

    Args:
        y_true       (np.ndarray): Ground-truth [n_samples, n_labels].
        y_pred_proba (np.ndarray): Predicted probabilities [n_samples, n_labels].
        label_cols   (list[str]) : Label names.
        save_path    (str|None)  : Where to save the threshold curve plot.

    Returns:
        dict: {label: best_threshold}
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_thresholds = {}
    all_f1s = {}

    for i, lbl in enumerate(label_cols):
        yt, yp = y_true[:, i], y_pred_proba[:, i]
        f1s = [f1_score(yt, (yp >= t).astype(int), zero_division=0)
               for t in thresholds]
        best_idx = int(np.argmax(f1s))
        best_t   = float(thresholds[best_idx])
        best_thresholds[lbl] = round(best_t, 2)
        all_f1s[lbl] = f1s
        print(f"    {lbl:<16}  best threshold = {best_t:.2f}  "
              f"F1 = {f1s[best_idx]:.4f}")

    # Plot curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ["#ff6b6b", "#ff9f43", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff"]
    for i, (lbl, f1s) in enumerate(all_f1s.items()):
        ax = axes[i]
        ax.plot(thresholds, f1s, color=colors[i], linewidth=2)
        bt = best_thresholds[lbl]
        ax.axvline(bt, color="#e6edf3", linewidth=1.2, linestyle="--",
                   label=f"best={bt:.2f}")
        ax.scatter([bt], [f1s[int(round((bt - 0.10) / 0.01))]],
                   color="#e6edf3", s=60, zorder=5)
        ax.set_title(lbl.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
        ax.set_xlim(0.10, 0.90); ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8); ax.grid(linestyle="--", linewidth=0.5)
    fig.suptitle("Per-Label Threshold vs F1 Score", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out = save_path or ppath("outputs/phase4/plots/threshold_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Threshold curves saved -> {out}")
    return best_thresholds


def apply_thresholds(proba, thresholds_dict, label_cols):
    """Convert probability matrix to binary predictions using per-label thresholds."""
    preds = np.zeros_like(proba, dtype=int)
    for i, lbl in enumerate(label_cols):
        preds[:, i] = (proba[:, i] >= thresholds_dict[lbl]).astype(int)
    return preds


def evaluate_predictions(y_true, y_pred, y_proba, label_cols, split_name="val"):
    """
    Compute and print macro F1 + per-label F1 scores.

    Args:
        y_true      (np.ndarray): Ground truth [n, 6].
        y_pred      (np.ndarray): Binary predictions [n, 6].
        y_proba     (np.ndarray): Raw probabilities [n, 6].
        label_cols  (list[str]) : Label names.
        split_name  (str)       : 'val' or 'test' for display.

    Returns:
        tuple: (f1_macro, per_label_f1_dict)
    """
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label = {}
    for i, lbl in enumerate(label_cols):
        per_label[lbl] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    print(f"\n  [{split_name.upper()}] F1 Macro = {f1_macro:.4f}")
    for lbl, score in per_label.items():
        print(f"    {lbl:<16}: {score:.4f}")
    return f1_macro, per_label


# =============================================================================
# INIT WANDB
# =============================================================================
if WANDB_OK:
    try:
        wandb.init(
            project="chat-toxicity-detector",
            config=config,
            mode=WANDB_MODE,      # "disabled" by default — no login needed
        )
        print(f"  wandb initialized (mode={WANDB_MODE}).")
    except Exception as e:
        print(f"  wandb init failed ({e}). Disabling wandb.")
        WANDB_OK = False


# =============================================================================
# DATA PREPARATION
# =============================================================================
section("DATA PREPARATION")

subsection("Loading train_augmented.csv")
t0 = time.time()
df = pd.read_csv(ppath(config["train_path"]))
print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
print(f"  Columns: {list(df.columns)}")

# Fill any NaN in text columns
df["comment_clean"] = df["comment_clean"].fillna("")
df["comment_light"] = df["comment_light"].fillna("")

# Verify label columns exist
for lbl in LABEL_COLS:
    assert lbl in df.columns, f"Missing label column: {lbl}"

# ── Train / Validation split ──────────────────────────────────────────────────
subsection("Train / Validation split (val_size=0.15)")
X_idx = np.arange(len(df))
y_mat = df[LABEL_COLS].values.astype(int)

if ITERSTRAT_OK:
    print("  Using MultilabelStratifiedShuffleSplit (iterstrat)")
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=config["val_size"],
        random_state=config["random_seed"],
    )
    train_idx, val_idx = next(msss.split(X_idx, y_mat))
else:
    print("  iterstrat not found — stratifying on 'toxic' column")
    train_idx, val_idx = train_test_split(
        X_idx,
        test_size=config["val_size"],
        random_state=config["random_seed"],
        stratify=df["toxic"].values,
    )

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)

print(f"  Train : {len(df_train):,} rows")
print(f"  Val   : {len(df_val):,} rows")

print(f"\n  Label distribution (positive rate %):")
print(f"  {'Label':<16} " + " ".join(f"{'Train':>8}" for _ in range(1)) +
      " " + " ".join(f"{'Val':>8}" for _ in range(1)))
print(f"  {'-'*40}")
for lbl in LABEL_COLS:
    tr_pct = df_train[lbl].mean() * 100
    vl_pct = df_val[lbl].mean() * 100
    print(f"  {lbl:<16}  train={tr_pct:.2f}%  val={vl_pct:.2f}%")

# ── Load test data ────────────────────────────────────────────────────────────
subsection("Loading test data")
df_test_raw = pd.read_csv(ppath(config["test_path"]))
df_test_lbl = pd.read_csv(ppath(config["test_labels_path"]))
df_test = df_test_raw.merge(df_test_lbl, on="id", how="inner")

# Filter out rows where all labels == -1 (unlabeled)
labeled_mask = (df_test[LABEL_COLS] != -1).any(axis=1)
df_test = df_test[labeled_mask].reset_index(drop=True)
df_test["comment_clean"] = df_test["comment_text"].fillna("")
df_test["comment_light"] = df_test["comment_text"].fillna("")
print(f"  Test set (labeled only): {len(df_test):,} rows")

y_train = df_train[LABEL_COLS].values.astype(int)
y_val   = df_val[LABEL_COLS].values.astype(int)
y_test  = df_test[LABEL_COLS].values.astype(int)

# Results tracking
results = {}
STAGE_TIMES = {}


# =============================================================================
#  STAGE 1: BASELINE MODELS (TF-IDF + LR + LightGBM)
# =============================================================================
if RUN_STAGE1:
    section("STAGE 1: BASELINE MODELS")
    t_stage1_start = time.time()



# =============================================================================
# 1A: TF-IDF VECTORIZATION
# =============================================================================
subsection("1A: TF-IDF Vectorization")

tfidf = TfidfVectorizer(
    max_features=config["tfidf_max_features"],
    ngram_range=config["tfidf_ngram_range"],
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    min_df=3,
)

t0 = time.time()
print("  Fitting TF-IDF on training set ...")
X_train_tfidf = tfidf.fit_transform(df_train["comment_clean"])
X_val_tfidf   = tfidf.transform(df_val["comment_clean"])
X_test_tfidf  = tfidf.transform(df_test["comment_clean"])
print(f"  TF-IDF fit+transform done in {time.time()-t0:.1f}s")
print(f"  Vocabulary size  : {len(tfidf.vocabulary_):,} tokens")
print(f"  Matrix shape     : train={X_train_tfidf.shape}, "
      f"val={X_val_tfidf.shape}, test={X_test_tfidf.shape}")

# Top 20 tokens by mean TF-IDF score
feature_names = np.array(tfidf.get_feature_names_out())
mean_tfidf = np.asarray(X_train_tfidf.mean(axis=0)).flatten()
top20_idx  = mean_tfidf.argsort()[::-1][:20]
print(f"  Top 20 tokens by mean TF-IDF score:")
print(f"  {list(feature_names[top20_idx])}")

# Save vectorizer
tfidf_path = ppath("outputs/phase4/saved_models/tfidf_vectorizer.pkl")
with open(tfidf_path, "wb") as f:
    pickle.dump(tfidf, f)
print(f"  TF-IDF vectorizer saved -> {tfidf_path}")


# =============================================================================
# 1B: LOGISTIC REGRESSION (One-vs-Rest)
# =============================================================================
subsection("1B: Logistic Regression (One-vs-Rest)")

t0 = time.time()
lr_model = OneVsRestClassifier(
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    n_jobs=-1,
)
print("  Training Logistic Regression (OvR) ...")
lr_model.fit(X_train_tfidf, y_train)
t_lr = time.time() - t0
print(f"  Trained in {t_lr:.1f}s")

# Probabilities on validation and test
lr_val_proba  = lr_model.predict_proba(X_val_tfidf)
lr_test_proba = lr_model.predict_proba(X_test_tfidf)

# Tune thresholds on validation probabilities
print("  Tuning thresholds on validation probabilities ...")
lr_thresholds = tune_thresholds(
    y_val, lr_val_proba, LABEL_COLS,
    save_path=ppath("outputs/phase4/plots/lr_threshold_curves.png"),
)

# Evaluate
lr_val_pred  = apply_thresholds(lr_val_proba,  lr_thresholds, LABEL_COLS)
lr_test_pred = apply_thresholds(lr_test_proba, lr_thresholds, LABEL_COLS)

lr_val_f1_macro, lr_val_per_label  = evaluate_predictions(
    y_val,  lr_val_pred,  lr_val_proba,  LABEL_COLS, "val")
lr_test_f1_macro, lr_test_per_label = evaluate_predictions(
    y_test, lr_test_pred, lr_test_proba, LABEL_COLS, "test")

results["logistic_regression"] = {
    "val_f1_macro":   lr_val_f1_macro,
    "test_f1_macro":  lr_test_f1_macro,
    "val_per_label":  lr_val_per_label,
    "test_per_label": lr_test_per_label,
    "train_time_s":   t_lr,
}

safe_log({
    "baseline_lr/val_f1_macro":  lr_val_f1_macro,
    "baseline_lr/test_f1_macro": lr_test_f1_macro,
    **{f"baseline_lr/val_f1_{lbl}": lr_val_per_label[lbl] for lbl in LABEL_COLS},
})

# Save model
lr_path = ppath("outputs/phase4/saved_models/logistic_regression.pkl")
with open(lr_path, "wb") as f:
    pickle.dump({"model": lr_model, "thresholds": lr_thresholds}, f)
print(f"  Saved -> outputs/phase4/saved_models/logistic_regression.pkl")


# =============================================================================
# 1C: LIGHTGBM (One-per-Label)
# =============================================================================
subsection("1C: LightGBM (One-per-Label)")

if LGBM_OK:
    lgbm_models    = {}
    lgbm_val_proba = np.zeros((len(df_val),  len(LABEL_COLS)))
    lgbm_test_proba = np.zeros((len(df_test), len(LABEL_COLS)))

    t0 = time.time()
    for i, lbl in enumerate(LABEL_COLS):
        print(f"\n  Training LightGBM for '{lbl}' "
              f"(pos_weight={class_weights_dict[lbl]:.2f}) ...")
        params = dict(
            objective="binary",
            metric="binary_logloss",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            scale_pos_weight=class_weights_dict[lbl],
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf = LGBMClassifier(**params)
        clf.fit(
            X_train_tfidf, y_train[:, i],
            eval_set=[(X_val_tfidf, y_val[:, i])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        lgbm_models[lbl] = clf
        lgbm_val_proba[:, i]  = clf.predict_proba(X_val_tfidf)[:, 1]
        lgbm_test_proba[:, i] = clf.predict_proba(X_test_tfidf)[:, 1]

        mdl_path = ppath(f"outputs/phase4/saved_models/lgbm_{lbl}.pkl")
        with open(mdl_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved -> outputs/phase4/saved_models/lgbm_{lbl}.pkl")

    t_lgbm = time.time() - t0
    print(f"\n  All LightGBM models trained in {t_lgbm:.1f}s")

    # Tune thresholds
    print("  Tuning thresholds on validation probabilities ...")
    lgbm_thresholds = tune_thresholds(
        y_val, lgbm_val_proba, LABEL_COLS,
        save_path=ppath("outputs/phase4/plots/lgbm_threshold_curves.png"),
    )

    # Evaluate
    lgbm_val_pred  = apply_thresholds(lgbm_val_proba,  lgbm_thresholds, LABEL_COLS)
    lgbm_test_pred = apply_thresholds(lgbm_test_proba, lgbm_thresholds, LABEL_COLS)

    lgbm_val_f1_macro, lgbm_val_per_label  = evaluate_predictions(
        y_val,  lgbm_val_pred,  lgbm_val_proba,  LABEL_COLS, "val")
    lgbm_test_f1_macro, lgbm_test_per_label = evaluate_predictions(
        y_test, lgbm_test_pred, lgbm_test_proba, LABEL_COLS, "test")

    results["lightgbm"] = {
        "val_f1_macro":   lgbm_val_f1_macro,
        "test_f1_macro":  lgbm_test_f1_macro,
        "val_per_label":  lgbm_val_per_label,
        "test_per_label": lgbm_test_per_label,
        "train_time_s":   t_lgbm,
    }

    safe_log({
        "baseline_lgbm/val_f1_macro":  lgbm_val_f1_macro,
        "baseline_lgbm/test_f1_macro": lgbm_test_f1_macro,
        **{f"baseline_lgbm/val_f1_{lbl}": lgbm_val_per_label[lbl]
           for lbl in LABEL_COLS},
    })

    # Feature importance plot for 'toxic'
    subsection("1C: Feature Importance (toxic label)")
    toxic_clf = lgbm_models["toxic"]
    importances = toxic_clf.feature_importances_
    top30_idx  = importances.argsort()[::-1][:30]
    top30_feat = feature_names[top30_idx]
    top30_imp  = importances[top30_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(30), top30_imp[::-1], color="#ff6b6b", edgecolor="#0d1117", linewidth=0.5)
    ax.set_yticks(range(30))
    ax.set_yticklabels(top30_feat[::-1], fontsize=9)
    ax.set_title("LightGBM Feature Importance (toxic label, top 30)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fi_path = ppath("outputs/phase4/plots/lgbm_feature_importance_toxic.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved -> {fi_path}")
else:
    print("  LightGBM not available — skipping 1C.")
    lgbm_val_f1_macro = lgbm_test_f1_macro = 0.0
    lgbm_val_per_label = lgbm_test_per_label = {lbl: 0.0 for lbl in LABEL_COLS}


# =============================================================================
# 1D: BASELINE EVALUATION SUMMARY
# =============================================================================
subsection("1D: Baseline Evaluation Summary")

W = [21, 10, 10, 11, 12, 11, 11, 12]
sep = "+" + "+".join("-" * w for w in W) + "+"
hdr = (f"| {'Model':<19}| {'F1 Macro':>8} | {'Toxic':>8} | "
       f"{'Severe':>9} | {'Obscene':>10} | {'Threat':>9} | "
       f"{'Insult':>9} | {'Id.Hate':>10} |")

print(f"\n  {sep}")
print(f"  {hdr}")
print(f"  {sep}")

def _row(name, f1m, per):
    cols = [name, f"{f1m:.4f}"] + [f"{per.get(l, 0):.4f}" for l in LABEL_COLS]
    row  = f"| {cols[0]:<19}|"
    row += f" {cols[1]:>8} |"
    row += f" {cols[2]:>8} |"
    row += f" {cols[3]:>9} |"
    row += f" {cols[4]:>10} |"
    row += f" {cols[5]:>9} |"
    row += f" {cols[6]:>9} |"
    row += f" {cols[7]:>10} |"
    return row

if RUN_STAGE1:
    print(f"  {_row('Logistic Regression', lr_val_f1_macro, lr_val_per_label)}")
    if LGBM_OK:
        print(f"  {_row('LightGBM', lgbm_val_f1_macro, lgbm_val_per_label)}")
    print(f"  {sep}")
    print("\n  (Validation set scores shown. BERT must beat these numbers.)\n")
    t_stage1_end = time.time()
    STAGE_TIMES["stage1_minutes"] = (t_stage1_end - t_stage1_start) / 60
    print(f"  Stage 1 completed in {STAGE_TIMES['stage1_minutes']:.1f} minutes")
else:
    section("STAGE 1: SKIPPED (--stage 2 was set)")
    lr_val_f1_macro    = lr_test_f1_macro    = 0.0
    lr_val_per_label   = lr_test_per_label   = {lbl: 0.0 for lbl in LABEL_COLS}
    lgbm_val_f1_macro  = lgbm_test_f1_macro  = 0.0
    lgbm_val_per_label = lgbm_test_per_label = {lbl: 0.0 for lbl in LABEL_COLS}


# ── Free Stage 1 RAM before loading BERT ──────────────────────────────────
import gc
if RUN_STAGE1 and RUN_STAGE2:
    print("\n  Freeing Stage 1 RAM before loading BERT ...")
    for _v in ["X_train_tfidf", "X_val_tfidf", "X_test_tfidf",
               "lr_model", "lgbm_models"]:
        if _v in dir():
            del globals()[_v]
    gc.collect()
    print("  RAM freed.")

# =============================================================================
#  STAGE 2: BERT FINE-TUNING
# =============================================================================
if not RUN_STAGE2:
    section("STAGE 2: SKIPPED (--stage 1 was set)")
elif not TORCH_OK:
    section("STAGE 2: BERT FINE-TUNING")
    print("  PyTorch not installed — skipping Stage 2.")
    print("  Run: pip install torch transformers")
else:
    section("STAGE 2: BERT FINE-TUNING")
    t_stage2_start = time.time()

    # ── Device detection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"  GPU memory before model load: {mem_before:.1f} MB")
        # GPU can handle larger batches/sequences
        config["max_length"]       = 256
        config["batch_size"]       = 16
        config["grad_accum_steps"] = 1
        print("  GPU mode: max_length=256, batch_size=16, grad_accum_steps=1")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU (memory-safe mode:")
        print(f"    max_length={config['max_length']}, "
              f"batch_size={config['batch_size']}, "
              f"grad_accum_steps={config['grad_accum_steps']})")

    # =========================================================================
    # 2A: DATASET CLASS
    # =========================================================================
    subsection("2A: ToxicCommentDataset")

    class ToxicCommentDataset(Dataset):
        """
        PyTorch Dataset for multi-label toxic comment classification.

        Uses lightly-cleaned text (comment_light) so BERT's WordPiece
        tokenizer can operate on near-original natural language.

        Args:
            texts      (list[str])    : Raw/lightly-cleaned comment strings.
            labels     (np.ndarray)   : Binary label matrix [n_samples, 6].
            tokenizer                 : HuggingFace tokenizer instance.
            max_length (int)          : Max token sequence length.
        """

        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts     = texts
            self.labels    = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            enc  = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.FloatTensor(self.labels[idx]),
            }

    # Load tokenizer
    subsection("Loading BERT tokenizer")
    print(f"  Loading tokenizer: {config['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    print("  Tokenizer loaded.")

    # Prepare text lists
    train_texts = df_train["comment_light"].tolist()
    val_texts   = df_val["comment_light"].tolist()
    test_texts  = df_test.get("comment_light",
                  df_test["comment_text"].fillna("")).tolist()
    _y_train_bert = y_train

    # ── Optional: cap training samples for CPU quick-runs ──────────────────
    bert_sample = config.get("bert_sample", None)
    if bert_sample is not None and bert_sample < len(train_texts):
        import random as _rnd
        _rnd.seed(config["random_seed"])
        _idx = _rnd.sample(range(len(train_texts)), bert_sample)
        _idx.sort()
        train_texts   = [train_texts[i]   for i in _idx]
        _y_train_bert = _y_train_bert[_idx]
        print(f"  [bert-sample] Training capped at {bert_sample:,} rows "
              f"(was {len(df_train):,}).")
    else:
        # Warn user about expected CPU time
        secs_per_batch = 3.5
        n_batches      = len(train_texts) / config["batch_size"]
        eta_hrs        = (secs_per_batch * n_batches * config["epochs"]) / 3600
        if device.type == "cpu" and eta_hrs > 1:
            print(f"\n  [WARNING] Estimated training time on CPU: {eta_hrs:.0f} hours.")
            print(f"  Tip: run with --bert-sample 5000 for a ~30-minute demo run instead.\n")

    # Datasets
    train_dataset = ToxicCommentDataset(train_texts, _y_train_bert, tokenizer, config["max_length"])
    val_dataset   = ToxicCommentDataset(val_texts,   y_val,         tokenizer, config["max_length"])
    test_dataset  = ToxicCommentDataset(test_texts,  y_test,        tokenizer, config["max_length"])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"] * 2,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"] * 2,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    print(f"  DataLoaders: train={len(train_loader)} batches, "
          f"val={len(val_loader)}, test={len(test_loader)}")

    # =========================================================================
    # 2B: MODEL ARCHITECTURE
    # =========================================================================
    subsection("2B: ToxicClassifier (BERT + linear head)")

    class ToxicClassifier(nn.Module):
        """
        BERT encoder with a 6-output linear classification head.

        Architecture:
          BERT base -> [CLS] token -> Dropout(0.3) -> Linear(768, 6)

        Outputs raw logits (NOT probabilities). Apply torch.sigmoid()
        at inference time. During training, BCEWithLogitsLoss combines
        sigmoid + BCE in one numerically stable operation.

        Args:
            model_name   (str)  : HuggingFace model identifier.
            num_labels   (int)  : Number of output labels (default 6).
            dropout_rate (float): Dropout probability (default 0.3).
        """

        def __init__(self, model_name, num_labels=6, dropout_rate=0.3):
            super().__init__()
            self.bert       = AutoModel.from_pretrained(model_name)
            self.dropout    = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            """
            Forward pass. Compatible with both BERT and DistilBERT.

            - BERT:       output.last_hidden_state[:, 0, :]  (CLS token)
            - DistilBERT: output.last_hidden_state[:, 0, :]  (same layout)

            Returns:
                Tensor: Raw logits [batch, num_labels]. Apply sigmoid for probabilities.
            """
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Works for both BERT and DistilBERT — both expose last_hidden_state
            cls    = out.last_hidden_state[:, 0, :]   # CLS token representation
            pooled = self.dropout(cls)
            return self.classifier(pooled)             # raw logits

    # Instantiate model
    print(f"  Loading {config['model_name']} weights ...")
    model = ToxicClassifier(config["model_name"], num_labels=len(LABEL_COLS))
    model = model.to(device)

    if device.type == "cuda":
        mem_after = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory after model load: {mem_after:.1f} MB")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params : {total_params:,}")
    print(f"  Trainable    : {trainable:,}")

    # =========================================================================
    # 2C: LOSS + 2D: OPTIMIZER + SCHEDULER
    # =========================================================================
    subsection("2C/D: Loss function, Optimizer, Scheduler")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
    print(f"  BCEWithLogitsLoss with pos_weight: {pos_weight_tensor.tolist()}")

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
        eps=1e-8,
    )

    total_steps   = len(train_loader) * config["epochs"]
    warmup_steps  = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  Total training steps : {total_steps:,}")
    print(f"  Warmup steps         : {warmup_steps:,}")

    # =========================================================================
    # 2E: TRAINING FUNCTIONS
    # =========================================================================

    def train_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
        """
        Run one training epoch over the full training DataLoader.

        Args:
            model     : ToxicClassifier instance.
            loader    : Training DataLoader.
            optimizer : AdamW optimiser.
            scheduler : Linear warmup + decay scheduler.
            criterion : BCEWithLogitsLoss with pos_weight.
            device    : torch.device.
            epoch     (int): Current epoch number (for display).

        Returns:
            float: Mean training loss for this epoch.
        """
        model.train()
        total_loss   = 0.0
        accum_steps  = config.get("grad_accum_steps", 1)
        pbar = tqdm(loader, desc=f"  Epoch {epoch} [TRAIN]",
                    unit="batch", ncols=90, leave=True)
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            # Scale loss by accum_steps so gradients average correctly
            loss   = criterion(logits, labels) / accum_steps
            loss.backward()

            total_loss += loss.item() * accum_steps  # log unscaled
            current_lr  = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}",
                             lr=f"{current_lr:.2e}")

            # Step optimizer every accum_steps batches OR on last batch
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        return total_loss / len(loader)


    def evaluate_bert(model, loader, criterion, device, label_cols,
                      thresholds=None):
        """
        Run inference over a DataLoader and compute F1 metrics.

        Args:
            model      : ToxicClassifier instance.
            loader     : Val or test DataLoader.
            criterion  : Loss function.
            device     : torch.device.
            label_cols (list[str]) : Label names.
            thresholds (dict|None) : Per-label thresholds. Uses 0.5 if None.

        Returns:
            tuple: (avg_loss, f1_macro, per_label_f1, all_proba, all_true)
        """
        model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="  [EVAL]",
                              unit="batch", ncols=90, leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_proba  = torch.sigmoid(all_logits).numpy()
        all_true   = all_labels.numpy().astype(int)

        if thresholds is None:
            thresholds = {lbl: 0.5 for lbl in label_cols}

        all_pred = apply_thresholds(all_proba, thresholds, label_cols)

        f1_mac = f1_score(all_true, all_pred, average="macro", zero_division=0)
        per_lbl = {lbl: f1_score(all_true[:, i], all_pred[:, i], zero_division=0)
                   for i, lbl in enumerate(label_cols)}

        avg_loss = total_loss / len(loader)
        return avg_loss, f1_mac, per_lbl, all_proba, all_true


    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    subsection("2E: Training Loop")

    best_f1_macro  = 0.0
    best_ckpt_path = None
    patience_count = 0
    early_stop_patience = 2

    for epoch in range(1, config["epochs"] + 1):
        t_ep = time.time()
        print(f"\n  {'='*60}")
        print(f"  EPOCH {epoch} / {config['epochs']}")
        print(f"  {'='*60}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch,
        )

        val_loss, val_f1_macro, val_per_label, _, _ = evaluate_bert(
            model, val_loader, criterion, device, LABEL_COLS,
        )

        ep_time = time.time() - t_ep
        print(f"\n  Epoch {epoch} summary:")
        print(f"    Train Loss : {train_loss:.4f}")
        print(f"    Val   Loss : {val_loss:.4f}")
        print(f"    Val F1 Mac : {val_f1_macro:.4f}")
        for lbl, sc in val_per_label.items():
            print(f"      {lbl:<16}: {sc:.4f}")
        print(f"    Epoch time : {ep_time:.1f}s")

        safe_log({
            "epoch":          epoch,
            "train/loss":     train_loss,
            "val/loss":       val_loss,
            "val/f1_macro":   val_f1_macro,
            "learning_rate":  scheduler.get_last_lr()[0],
            **{f"val/f1_{lbl}": val_per_label[lbl] for lbl in LABEL_COLS},
        })

        # Save best checkpoint
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            patience_count = 0
            ckpt_name = f"bert_epoch{epoch}_f1{val_f1_macro:.4f}.pt"
            best_ckpt_path = ppath(f"outputs/phase4/saved_models/{ckpt_name}")
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "val_f1_macro":        val_f1_macro,
                "val_per_label_f1":    val_per_label,
                "config":              config,
            }, best_ckpt_path)
            print(f"  ** New best! Checkpoint saved -> {ckpt_name}")
        else:
            patience_count += 1
            print(f"  No improvement. Patience: {patience_count}/{early_stop_patience}")
            if patience_count >= early_stop_patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    # =========================================================================
    # 2F: POST-TRAINING THRESHOLD TUNING
    # =========================================================================
    subsection("2F: Post-Training Threshold Tuning")

    if best_ckpt_path and os.path.exists(best_ckpt_path):
        print(f"  Loading best checkpoint: {os.path.basename(best_ckpt_path)}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Best epoch: {ckpt['epoch']}  |  Best val F1: {ckpt['val_f1_macro']:.4f}")
    else:
        print("  No checkpoint found — using current model weights.")

    # Collect validation probabilities
    _, _, _, bert_val_proba, bert_val_true = evaluate_bert(
        model, val_loader, criterion, device, LABEL_COLS,
    )

    # Tune thresholds on real validation probabilities
    print("\n  Tuning thresholds on real BERT validation probabilities ...")
    bert_thresholds = tune_thresholds(
        bert_val_true, bert_val_proba, LABEL_COLS,
        save_path=ppath("outputs/phase4/plots/bert_threshold_curves.png"),
    )

    # Save tuned thresholds
    thresh_path = ppath("outputs/phase4/tuned_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(bert_thresholds, f, indent=2)
    print(f"  Tuned thresholds saved -> outputs/phase4/tuned_thresholds.json")

    # Re-evaluate validation with tuned thresholds
    val_loss_t, bert_val_f1_tuned, bert_val_per_t, _, _ = evaluate_bert(
        model, val_loader, criterion, device, LABEL_COLS, bert_thresholds,
    )

    # Evaluate test set with tuned thresholds
    test_loss_t, bert_test_f1, bert_test_per, bert_test_proba, bert_test_true = \
        evaluate_bert(model, test_loader, criterion, device, LABEL_COLS,
                      bert_thresholds)

    print(f"\n  BERT Results with Tuned Thresholds:")
    print(f"  {'Split':<10}  {'F1 Macro':>9}  " +
          "  ".join(f"{l:>12}" for l in LABEL_COLS))
    print(f"  {'-'*100}")

    def _result_row(split, f1m, per):
        return (f"  {split:<10}  {f1m:>9.4f}  " +
                "  ".join(f"{per.get(l,0):>12.4f}" for l in LABEL_COLS))

    print(_result_row("Validation", bert_val_f1_tuned, bert_val_per_t))
    print(_result_row("Test",       bert_test_f1,      bert_test_per))

    results["bert"] = {
        "val_f1_macro":   bert_val_f1_tuned,
        "test_f1_macro":  bert_test_f1,
        "val_per_label":  bert_val_per_t,
        "test_per_label": bert_test_per,
        "train_time_s":   time.time() - t_stage2_start,
    }

    safe_log({
        "bert/val_f1_macro_tuned":  bert_val_f1_tuned,
        "bert/test_f1_macro":       bert_test_f1,
        **{f"bert/test_f1_{lbl}": bert_test_per[lbl] for lbl in LABEL_COLS},
    })

    t_stage2_end = time.time()
    STAGE_TIMES["stage2_minutes"] = (t_stage2_end - t_stage2_start) / 60
    print(f"\n  Stage 2 completed in {STAGE_TIMES['stage2_minutes']:.1f} minutes")


# =============================================================================
# FULL RESULTS COMPARISON
# =============================================================================
section("FULL RESULTS COMPARISON")

print(f"\n  {'Model':<22} {'Val F1':>9} {'Test F1':>9} {'Time (min)':>12}")
print(f"  {'-'*56}")

model_results = [
    ("Logistic Regression",
     results.get("logistic_regression", {}).get("val_f1_macro",  0),
     results.get("logistic_regression", {}).get("test_f1_macro", 0),
     results.get("logistic_regression", {}).get("train_time_s",  0) / 60),
    ("LightGBM",
     results.get("lightgbm", {}).get("val_f1_macro",  0),
     results.get("lightgbm", {}).get("test_f1_macro", 0),
     results.get("lightgbm", {}).get("train_time_s",  0) / 60),
    ("BERT fine-tuned",
     results.get("bert", {}).get("val_f1_macro",  0),
     results.get("bert", {}).get("test_f1_macro", 0),
     results.get("bert", {}).get("train_time_s",  0) / 60),
]

for name, val_f1, test_f1, mins in model_results:
    print(f"  {name:<22} {val_f1:>9.4f} {test_f1:>9.4f} {mins:>11.1f}m")

# Log final table to wandb
if WANDB_OK:
    tbl = wandb.Table(
        columns=["Model", "Val F1 Macro", "Test F1 Macro", "Training Time (min)"],
        data=[[n, f"{v:.4f}", f"{t:.4f}", f"{m:.1f}"]
              for n, v, t, m in model_results],
    )
    wandb.log({"final_comparison": tbl})
    wandb.finish()
    print("  wandb run finished.")


# =============================================================================
# SAVE PHASE 4 SUMMARY JSON
# =============================================================================
section("SAVING PHASE 4 SUMMARY")

summary = {
    "phase": 4,
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "stage_times":  STAGE_TIMES,
    "results":      {
        model: {
            "val_f1_macro":   float(info.get("val_f1_macro", 0)),
            "test_f1_macro":  float(info.get("test_f1_macro", 0)),
            "train_time_min": float(info.get("train_time_s", 0)) / 60,
            "val_per_label":  {k: float(v)
                               for k, v in info.get("val_per_label", {}).items()},
            "test_per_label": {k: float(v)
                               for k, v in info.get("test_per_label", {}).items()},
        }
        for model, info in results.items()
    },
    "config": {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in config.items()
    },
    "class_weights": class_weights_dict,
}

summary_path = ppath("outputs/phase4/phase4_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved -> outputs/phase4/phase4_summary.json")

# Final outputs listing
print(f"\n  Files written to outputs/phase4/:")
for root, dirs, files in os.walk(ppath("outputs/phase4")):
    dirs[:] = sorted(dirs)
    for fname in sorted(files):
        full = os.path.join(root, fname)
        rel  = os.path.relpath(full, ppath("outputs/phase4"))
        sz   = os.path.getsize(full)
        print(f"    {rel:<55} {sz/1024:>8.1f} KB")

print("\n  Phase 4 complete.")
