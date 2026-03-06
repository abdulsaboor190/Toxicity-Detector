# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
=============================================================================
  PHASE 3 - Preprocessing Pipeline & Class-Imbalance Handling
  Chat Toxicity Detector | Jigsaw Toxic Comment Classification
=============================================================================
  JOB 1 : Text Preprocessing (aggressive + light cleaning)
  JOB 2 : Class Imbalance  (weights + threshold tuning + augmentation)
=============================================================================
"""

import os
import re
import json
import time
import random
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import nltk
import spacy
import contractions

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ── Seeds ────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "phase3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MATPLOTLIB_DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "savefig.facecolor": "#0d1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "axes.edgecolor": "#30363d",
    "grid.color": "#21262d", "font.family": "DejaVu Sans",
    "axes.titlecolor": "#e6edf3",
}
matplotlib.rcParams.update(MATPLOTLIB_DARK)


# ── Pretty printing helpers ───────────────────────────────────────────────────
def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n-- {title} " + "-" * max(0, 65 - len(title)))

def tick(label, status="PASS"):
    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"  {icon}  {label}")


# =============================================================================
# INITIALISE SHARED RESOURCES (loaded once, reused everywhere)
# =============================================================================
section("INITIALISING SHARED RESOURCES")

print("  Loading spaCy en_core_web_sm ...")
t0 = time.time()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print(f"  spaCy loaded in {time.time()-t0:.1f}s")

print("  Downloading NLTK stopwords ...")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_sw

CUSTOM_STOPWORDS = {
    "wikipedia", "article", "page", "talk", "edit",
    "user", "would", "also",
}
ALL_STOPWORDS = set(nltk_sw.words("english")) | CUSTOM_STOPWORDS
print(f"  Stopword list size: {len(ALL_STOPWORDS)} tokens")


# =============================================================================
#  ██╗ ██████╗ ██████╗      ██╗
#  ██║██╔═══██╗██╔══██╗    ███║
#  ██║██║   ██║██████╔╝    ╚██║
#  ██║██║   ██║██╔══██╗     ██║
#  ██║╚██████╔╝██████╔╝     ██║
#  ╚═╝ ╚═════╝ ╚═════╝      ╚═╝
#   JOB 1: TEXT PREPROCESSING PIPELINE
# =============================================================================
section("JOB 1: TEXT PREPROCESSING PIPELINE")


# ── Compiled regex patterns (compile once for speed) ─────────────────────────
RE_URL     = re.compile(r"https?://\S+|www\.\S+")
RE_IP      = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
RE_HTML    = re.compile(r"<[^>]+>")
RE_REPEAT  = re.compile(r"(.)\1{2,}")
RE_SPECIAL = re.compile(r"[^a-z0-9\s']")
RE_SPACES  = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE A: Aggressive Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text_aggressive(text: str) -> str:
    """
    Apply 12-step aggressive text cleaning optimised for TF-IDF / classical ML.

    Steps (in order):
      1.  Lowercase
      2.  Remove URLs
      3.  Remove IP addresses
      4.  Expand contractions  ("don't" -> "do not")
      5.  Strip HTML tags
      6.  Normalise repeated chars (3+ -> 2)
      7.  Remove special characters (keep letters, digits, spaces, apostrophes)
      8.  Collapse whitespace
      9.  spaCy tokenisation
      10. Remove stopwords (NLTK + custom domain list)
      11. Lemmatise with spaCy .lemma_
      12. Rejoin tokens

    Args:
        text (str): Raw comment string.

    Returns:
        str: Cleaned, lemmatised string. Empty string if all tokens are removed.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1 – lowercase
    text = text.lower()
    # Step 2 – remove URLs
    text = RE_URL.sub("", text)
    # Step 3 – remove IP addresses
    text = RE_IP.sub("", text)
    # Step 4 – expand contractions
    try:
        text = contractions.fix(text)
    except Exception:
        pass  # contractions library occasionally fails on edge-case strings
    # Step 5 – strip HTML tags
    text = RE_HTML.sub("", text)
    # Step 6 – normalise repeated characters  sooooo -> soo
    text = RE_REPEAT.sub(r"\1\1", text)
    # Step 7 – keep only letters, digits, spaces, apostrophes
    text = RE_SPECIAL.sub(" ", text)
    # Step 8 – collapse whitespace
    text = RE_SPACES.sub(" ", text).strip()

    if not text:
        return ""

    # Steps 9-12 – spaCy tokenise, stopword removal, lemmatise, rejoin
    doc = nlp(text)
    tokens = [
        tok.lemma_
        for tok in doc
        if tok.text not in ALL_STOPWORDS
        and tok.lemma_ not in ALL_STOPWORDS
        and not tok.is_space
        and len(tok.text) > 1
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE B: Light Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text_light(text: str) -> str:
    """
    Apply minimal cleaning suitable for BERT / transformer models.

    BERT's own WordPiece tokeniser handles lowercasing, punctuation, and
    subword splitting. Over-cleaning degrades its pretrained representations.

    Steps:
      1. Replace URLs with [URL] placeholder
      2. Strip HTML tags
      3. Normalise repeated characters (3+ -> 2)
      4. Decode unicode escape sequences
      5. Strip leading/trailing whitespace

    Args:
        text (str): Raw comment string.

    Returns:
        str: Lightly cleaned string preserving case, punctuation, and structure.
    """
    if not isinstance(text, str):
        return ""

    # Step 1 – replace URLs with placeholder token
    text = RE_URL.sub("[URL]", text)
    # Step 2 – remove HTML tags
    text = RE_HTML.sub("", text)
    # Step 3 – normalise repeated characters
    text = RE_REPEAT.sub(r"\1\1", text)
    # Step 4 – fix unicode escape sequences
    try:
        text = text.encode("utf-8").decode("unicode_escape", errors="replace")
    except Exception:
        try:
            text = text.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            pass
    # Step 5 – strip whitespace
    text = text.strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# BATCH APPLICATION  (uses nlp.pipe for spaCy efficiency)
# ─────────────────────────────────────────────────────────────────────────────
def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply both cleaning pipelines to every row of the dataframe.

    Uses spaCy's nlp.pipe() for batch processing the expensive NLP steps,
    which is significantly faster than calling nlp() per row.

    Args:
        df (pd.DataFrame): Must contain a 'comment_text' column.

    Returns:
        pd.DataFrame: Same dataframe with two new columns added:
            - comment_clean : aggressively cleaned text
            - comment_light : lightly cleaned text
    """
    texts = df["comment_text"].fillna("").tolist()
    n     = len(texts)

    # -- Light cleaning (pure regex, fast) ------------------------------------
    subsection("Applying light cleaning (Pipeline B)")
    t_start = time.time()
    light_results = []
    for txt in tqdm(texts, desc="  Light clean", unit="rows", ncols=80):
        light_results.append(clean_text_light(txt))
    print(f"  Done in {time.time()-t_start:.1f}s")

    # -- Aggressive cleaning: regex pre-pass first (no spaCy yet) -------------
    subsection("Aggressive cleaning: regex pre-pass (steps 1-8)")
    t_start = time.time()
    pre_texts = []
    for txt in tqdm(texts, desc="  Regex pass ", unit="rows", ncols=80):
        t = txt.lower() if isinstance(txt, str) else ""
        t = RE_URL.sub("", t)
        t = RE_IP.sub("", t)
        try:
            t = contractions.fix(t)
        except Exception:
            pass
        t = RE_HTML.sub("", t)
        t = RE_REPEAT.sub(r"\1\1", t)
        t = RE_SPECIAL.sub(" ", t)
        t = RE_SPACES.sub(" ", t).strip()
        pre_texts.append(t)
    print(f"  Regex pre-pass done in {time.time()-t_start:.1f}s")

    # -- Aggressive cleaning: spaCy batch (steps 9-12) ------------------------
    subsection("Aggressive cleaning: spaCy batch lemmatise (steps 9-12)")
    t_start = time.time()
    clean_results = []
    batch_size = 512
    with tqdm(total=n, desc="  spaCy pipe ", unit="rows", ncols=80) as pbar:
        for doc in nlp.pipe(pre_texts, batch_size=batch_size):
            tokens = [
                tok.lemma_
                for tok in doc
                if tok.text not in ALL_STOPWORDS
                and tok.lemma_ not in ALL_STOPWORDS
                and not tok.is_space
                and len(tok.text) > 1
            ]
            clean_results.append(" ".join(tokens))
            pbar.update(1)
    print(f"  spaCy batch done in {time.time()-t_start:.1f}s")

    df = df.copy()
    df["comment_clean"] = clean_results
    df["comment_light"] = light_results
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & APPLY
# ─────────────────────────────────────────────────────────────────────────────
subsection("Loading train.csv")
t0 = time.time()
train_path = os.path.join(DATA_DIR, "train.csv")
df = pd.read_csv(train_path)
print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
print(f"  Columns: {list(df.columns)}")

# Store original word count for comparison
df["_orig_word_count"] = df["comment_text"].fillna("").str.split().str.len()

t_total = time.time()
df = apply_preprocessing(df)
print(f"\n  Total preprocessing time: {time.time()-t_total:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION CHECKS
# ─────────────────────────────────────────────────────────────────────────────
subsection("Validation checks")

# 1. Empty rows after aggressive cleaning
empty_mask = (df["comment_clean"] == "") | (df["comment_clean"].isna())
n_empty = empty_mask.sum()
print(f"\n  [1] Rows with empty comment_clean  : {n_empty:,}")
if n_empty > 0:
    print(f"      (Flagged — these rows will likely contribute no signal to TF-IDF)")
    print(f"      Sample IDs: {df[empty_mask]['id'].head(5).tolist()}")

# 2. Average token count before vs after
df["_clean_word_count"] = df["comment_clean"].str.split().str.len().fillna(0)
avg_before = df["_orig_word_count"].mean()
avg_after  = df["_clean_word_count"].mean()
reduction  = (1 - avg_after / avg_before) * 100
print(f"\n  [2] Average word count")
print(f"      Before aggressive cleaning : {avg_before:.1f} words/comment")
print(f"      After  aggressive cleaning : {avg_after:.1f} words/comment")
print(f"      Vocabulary reduction       : {reduction:.1f}%")

# 3. NaN check
nan_clean = df["comment_clean"].isna().sum()
nan_light = df["comment_light"].isna().sum()
print(f"\n  [3] NaN in comment_clean : {nan_clean}")
print(f"      NaN in comment_light  : {nan_light}")

# 4. Side-by-side examples
print(f"\n  [4] Sample side-by-side comparison (5 rows)")
print(f"  {'='*90}")
sample_indices = df[df["comment_clean"] != ""].sample(5, random_state=42).index
for i, idx in enumerate(sample_indices, 1):
    row = df.loc[idx]
    orig  = str(row["comment_text"])[:120].replace("\n", " ")
    clean = str(row["comment_clean"])[:120]
    light = str(row["comment_light"])[:120]
    print(f"\n  Example {i}:")
    print(f"    ORIGINAL : {orig}")
    print(f"    AGGRESSIVE: {clean}")
    print(f"    LIGHT    : {light}")
print(f"  {'='*90}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE PROCESSED CSV
# ─────────────────────────────────────────────────────────────────────────────
subsection("Saving train_processed.csv")
drop_cols = ["_orig_word_count", "_clean_word_count"]
save_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
processed_path = os.path.join(OUTPUT_DIR, "train_processed.csv")
save_df.to_csv(processed_path, index=False)
print(f"  Saved to outputs/phase3/train_processed.csv [checkmark]")
print(f"  Shape: {save_df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE STANDALONE PREPROCESSOR MODULE
# ─────────────────────────────────────────────────────────────────────────────
subsection("Writing outputs/phase3/preprocessor.py")

PREPROCESSOR_CODE = r'''# -*- coding: utf-8 -*-
"""
preprocessor.py
---------------
Standalone preprocessing module for the Chat Toxicity Detector.

Usage:
    from preprocessor import clean_text_aggressive, clean_text_light

Both functions are safe to call on single strings.
For bulk processing use nlp.pipe() externally for speed.
"""

import re
import warnings
import contractions
import spacy
import nltk

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_sw

# -- Load spaCy model (once at import time) ----------------------------------
try:
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

_CUSTOM_SW = {
    "wikipedia", "article", "page", "talk", "edit",
    "user", "would", "also",
}
_ALL_SW = set(nltk_sw.words("english")) | _CUSTOM_SW

# -- Compiled patterns -------------------------------------------------------
_RE_URL     = re.compile(r"https?://\S+|www\.\S+")
_RE_IP      = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_HTML    = re.compile(r"<[^>]+>")
_RE_REPEAT  = re.compile(r"(.)\1{2,}")
_RE_SPECIAL = re.compile(r"[^a-z0-9\s']")
_RE_SPACES  = re.compile(r"\s+")


def clean_text_aggressive(text: str) -> str:
    """
    12-step aggressive cleaning pipeline for TF-IDF / classical ML.

    Args:
        text (str): Raw comment text.

    Returns:
        str: Lowercased, lemmatised, stopword-free string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _RE_URL.sub("", text)
    text = _RE_IP.sub("", text)
    try:
        text = contractions.fix(text)
    except Exception:
        pass
    text = _RE_HTML.sub("", text)
    text = _RE_REPEAT.sub(r"\1\1", text)
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()
    if not text:
        return ""
    doc = _nlp(text)
    tokens = [
        tok.lemma_
        for tok in doc
        if tok.text not in _ALL_SW
        and tok.lemma_ not in _ALL_SW
        and not tok.is_space
        and len(tok.text) > 1
    ]
    return " ".join(tokens)


def clean_text_light(text: str) -> str:
    """
    5-step light cleaning pipeline for BERT / transformer models.

    Args:
        text (str): Raw comment text.

    Returns:
        str: URL-replaced, HTML-stripped string preserving natural language.
    """
    if not isinstance(text, str):
        return ""
    text = _RE_URL.sub("[URL]", text)
    text = _RE_HTML.sub("", text)
    text = _RE_REPEAT.sub(r"\1\1", text)
    try:
        text = text.encode("utf-8").decode("unicode_escape", errors="replace")
    except Exception:
        try:
            text = text.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            pass
    return text.strip()
'''

preprocessor_path = os.path.join(OUTPUT_DIR, "preprocessor.py")
with open(preprocessor_path, "w", encoding="utf-8") as f:
    f.write(PREPROCESSOR_CODE)
print(f"  Saved to outputs/phase3/preprocessor.py [checkmark]")

# Quick import smoke-test
import importlib.util, sys as _sys
spec = importlib.util.spec_from_file_location("preprocessor", preprocessor_path)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
assert hasattr(_mod, "clean_text_aggressive") and hasattr(_mod, "clean_text_light")
print("  Import smoke-test: PASSED")


# =============================================================================
#  ██╗ ██████╗ ██████╗     ██████╗
#  ██║██╔═══██╗██╔══██╗    ╚════██╗
#  ██║██║   ██║██████╔╝     █████╔╝
#  ██║██║   ██║██╔══██╗    ██╔═══╝
#  ██║╚██████╔╝██████╔╝    ███████╗
#  ╚═╝ ╚═════╝ ╚═════╝     ╚══════╝
#   JOB 2: HANDLING CLASS IMBALANCE
# =============================================================================
section("JOB 2: HANDLING CLASS IMBALANCE")


# =============================================================================
# STRATEGY 1: Class Weights Computation
# =============================================================================
subsection("STRATEGY 1 -- Class Weights")

def compute_class_weights(df: pd.DataFrame, label_cols: list) -> dict:
    """
    Compute per-label class weights to address class imbalance in the loss function.

    Two methods are computed and printed:
      - Custom formula : weight = total / (2 * positives)
        Intuition: gives higher weight to rare classes; roughly balances the
        gradient contribution of each class.
      - sklearn's compute_class_weight('balanced') for verification.

    Args:
        df         (pd.DataFrame): Processed dataframe containing label columns.
        label_cols (list[str])   : Names of the binary label columns.

    Returns:
        dict: {label: float_weight} using the custom formula.
              These weights are passed to the loss function in Phase 4.
    """
    total = len(df)
    weights = {}

    print(f"\n  {'Label':<16} {'Pos':>7} {'Pos%':>7} {'Weight':>10}  Interpretation")
    print(f"  {'-'*75}")

    for lbl in label_cols:
        pos = int(df[lbl].sum())
        pct = pos / total * 100

        # Custom formula
        w_custom = total / (2 * pos) if pos > 0 else 1.0

        # sklearn balanced
        y = df[lbl].values
        w_sk = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        w_sklearn_pos = w_sk[1]

        weights[lbl] = round(w_custom, 4)
        interp = f"a missed {lbl} = {w_custom:.0f}x more costly than a clean comment"
        print(f"  {lbl:<16} {pos:>7,} {pct:>6.2f}%  {w_custom:>10.2f}  {interp}")

    print(f"\n  Note: sklearn balanced weights (for comparison):")
    for lbl in label_cols:
        y   = df[lbl].values
        w_s = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        print(f"    {lbl:<16}  pos_weight = {w_s[1]:.4f}")

    return weights


t0 = time.time()
class_weights = compute_class_weights(df, LABEL_COLS)
print(f"\n  Computed in {time.time()-t0:.2f}s")

weights_path = os.path.join(OUTPUT_DIR, "class_weights.json")
with open(weights_path, "w") as f:
    json.dump(class_weights, f, indent=2)
print(f"  Saved to outputs/phase3/class_weights.json [checkmark]")


# =============================================================================
# STRATEGY 2: Per-Label Threshold Tuning Framework
# =============================================================================
subsection("STRATEGY 2 -- Per-Label Threshold Tuning Framework")

def tune_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_cols: list,
    save_path: str = None,
) -> dict:
    """
    Find the probability threshold that maximises F1 for each label independently.

    For imbalanced datasets, the default 0.5 threshold is often suboptimal.
    Rare labels (like 'threat') benefit from lower thresholds so the model
    is more willing to flag potentially toxic comments.

    Algorithm:
      For each label:
        1. Sweep threshold t from 0.10 to 0.90 in steps of 0.01
        2. Compute F1 at each t (zero_division=0 to avoid warnings)
        3. Select t* = argmax(F1)

    Args:
        y_true       (np.ndarray): Ground-truth binary matrix [n_samples, n_labels].
        y_pred_proba (np.ndarray): Predicted probabilities    [n_samples, n_labels].
        label_cols   (list[str]) : Label names for each column.
        save_path    (str|None)  : If given, saves the threshold-F1 curve plot there.

    Returns:
        dict: {label: best_threshold (float)}
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_thresholds = {}
    all_f1s = {}

    for i, lbl in enumerate(label_cols):
        yt = y_true[:, i]
        yp = y_pred_proba[:, i]
        f1s = []
        for t in thresholds:
            pred = (yp >= t).astype(int)
            f1   = f1_score(yt, pred, zero_division=0)
            f1s.append(f1)
        best_idx = int(np.argmax(f1s))
        best_t   = float(thresholds[best_idx])
        best_f1  = f1s[best_idx]
        best_thresholds[lbl] = round(best_t, 2)
        all_f1s[lbl] = f1s
        print(f"    {lbl:<16}  best threshold = {best_t:.2f}  |  F1 = {best_f1:.4f}")

    # -- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    label_colors = ["#ff6b6b", "#ff9f43", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff"]

    for i, (lbl, f1s) in enumerate(all_f1s.items()):
        ax  = axes[i]
        col = label_colors[i]
        ax.plot(thresholds, f1s, color=col, linewidth=2)
        best_t = best_thresholds[lbl]
        best_f = f1s[int(round((best_t - 0.10) / 0.01))]
        ax.axvline(best_t, color="#e6edf3", linewidth=1.2, linestyle="--",
                   label=f"best = {best_t:.2f}")
        ax.scatter([best_t], [best_f], color="#e6edf3", s=60, zorder=5)
        ax.set_title(lbl.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.set_ylabel("F1 Score", fontsize=9)
        ax.set_xlim(0.10, 0.90)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", linewidth=0.5)

    fig.suptitle(
        "Per-Label Threshold vs F1 Score\n(sweep range: 0.10 - 0.90)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = save_path or os.path.join(OUTPUT_DIR, "threshold_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to outputs/phase3/threshold_curves.png [checkmark]")

    return best_thresholds


print("\n  Testing tune_thresholds() on dummy data (1000 samples) ...")
dummy_proba = np.random.uniform(0, 1, size=(1000, 6))
dummy_true  = (dummy_proba > 0.7).astype(int)

t0 = time.time()
best_thresholds = tune_thresholds(
    dummy_true, dummy_proba, LABEL_COLS,
    save_path=os.path.join(OUTPUT_DIR, "threshold_curves.png"),
)
print(f"  tune_thresholds() completed in {time.time()-t0:.2f}s")
print(f"  Best thresholds: {best_thresholds}")


# =============================================================================
# STRATEGY 3: Data Augmentation via Back-Translation
# =============================================================================
subsection("STRATEGY 3 -- Back-Translation Augmentation")

def augment_minority_labels(
    df: pd.DataFrame,
    label: str,
    n_samples: int,
    src_lang: str = "en",
    mid_lang: str = "fr",
    text_col: str = "comment_light",
) -> pd.DataFrame:
    """
    Generate synthetic training examples for a minority class via back-translation.

    Process:
      1. Filter df to rows where `label` == 1
      2. Sample up to n_samples rows
      3. Translate each comment: English -> French -> English
         (using Helsinki-NLP MarianMT models via nlpaug)
      4. Apply light cleaning to the result
      5. Copy all original label columns to the new synthetic row

    Falls back to BERT contextual word embedding augmentation if Helsinki
    models are unavailable (network / HuggingFace connectivity issues).

    Args:
        df        (pd.DataFrame) : Processed training dataframe.
        label     (str)          : Target label column (e.g. 'threat').
        n_samples (int)          : Max synthetic samples to generate.
        src_lang  (str)          : Source language code (default 'en').
        mid_lang  (str)          : Pivot language code (default 'fr').
        text_col  (str)          : Column to augment (default 'comment_light').

    Returns:
        pd.DataFrame: Rows of synthetic augmented samples WITH all label columns.
    """
    minority_df = df[df[label] == 1].copy()
    n_available = len(minority_df)
    sample_size = min(n_samples, n_available)
    sample_df   = minority_df.sample(sample_size, random_state=42)

    print(f"\n  Augmenting '{label}': {n_available} available -> targeting {sample_size} new samples")

    # -- Build augmenter: try back-translation, fall back to BERT ContextAug --
    aug = None
    use_backtrans = True
    try:
        import nlpaug.augmenter.word as naw
        print(f"  Attempting to load Helsinki-NLP MarianMT (en->fr->en) ...")
        aug = naw.BackTranslationAug(
            from_model_name=f"Helsinki-NLP/opus-mt-{src_lang}-{mid_lang}",
            to_model_name  =f"Helsinki-NLP/opus-mt-{mid_lang}-{src_lang}",
            device="cpu",
            batch_size=8,
        )
        # Smoke test with one short string
        _ = aug.augment(["hello world"])
        print(f"  Helsinki model loaded successfully.")
    except Exception as e:
        use_backtrans = False
        print(f"  Helsinki model unavailable ({type(e).__name__}: {str(e)[:80]})")
        print(f"  Falling back to BERT ContextualWordEmbsAug ...")
        try:
            import nlpaug.augmenter.word as naw
            aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased",
                action="substitute",
                device="cpu",
                batch_size=8,
            )
            _ = aug.augment(["hello world"])
            print(f"  BERT ContextualWordEmbsAug loaded successfully.")
        except Exception as e2:
            print(f"  Fallback also failed ({type(e2).__name__}). "
                  f"Using simple word-drop augmentation as last resort.")
            aug = None

    # -- Augmentation loop ----------------------------------------------------
    augmented_rows = []
    succeeded = 0
    failed    = 0

    texts    = sample_df[text_col].fillna("").tolist()
    label_data = sample_df[LABEL_COLS].values

    for idx, (text, labels_vec) in enumerate(
        tqdm(zip(texts, label_data),
             total=len(texts),
             desc=f"  Augmenting {label}",
             unit="comments",
             ncols=80)
    ):
        try:
            if aug is not None:
                augmented_list = aug.augment(text[:512])  # cap length for speed
                aug_text = augmented_list[0] if augmented_list else text
            else:
                # Last-resort: randomly drop 20% of words
                words = text.split()
                keep  = [w for w in words if random.random() > 0.20]
                aug_text = " ".join(keep) if keep else text

            aug_text = clean_text_light(str(aug_text))

            row = {text_col: aug_text, "comment_clean": "", "id": f"aug_{label}_{idx}"}
            for col, val in zip(LABEL_COLS, labels_vec):
                row[col] = int(val)
            augmented_rows.append(row)
            succeeded += 1

        except Exception:
            failed += 1
            continue

    print(f"  Results: {succeeded} succeeded | {failed} failed")

    if not augmented_rows:
        return pd.DataFrame()

    aug_df = pd.DataFrame(augmented_rows)
    # Fill any missing columns from original df structure
    for col in df.columns:
        if col not in aug_df.columns:
            aug_df[col] = np.nan
    aug_df = aug_df[df.columns.tolist()]
    return aug_df


# ── Run augmentation ─────────────────────────────────────────────────────────
print("\n  Before augmentation:")
for lbl in ["threat", "identity_hate"]:
    count = int(df[lbl].sum())
    print(f"    {lbl:<16}: {count:,} positive samples")

t0 = time.time()
aug_threat = augment_minority_labels(df, label="threat",        n_samples=500)
aug_ihate  = augment_minority_labels(df, label="identity_hate", n_samples=500)
print(f"\n  Augmentation total time: {time.time()-t0:.1f}s")

# ── Combine and shuffle ───────────────────────────────────────────────────────
df_augmented = pd.concat([df, aug_threat, aug_ihate], ignore_index=True)
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n  After augmentation:")
for lbl in ["threat", "identity_hate"]:
    count = int(df_augmented[lbl].sum())
    print(f"    {lbl:<16}: {count:,} positive samples")

print(f"\n  Total rows: {len(df):,} -> {len(df_augmented):,} "
      f"(+{len(df_augmented)-len(df):,} synthetic)")

augmented_path = os.path.join(OUTPUT_DIR, "train_augmented.csv")
df_augmented.to_csv(augmented_path, index=False)
print(f"  Saved to outputs/phase3/train_augmented.csv [checkmark]")


# =============================================================================
# INTEGRATION CHECK
# =============================================================================
section("INTEGRATION CHECK")

checks = {}

# [1] train_processed.csv exists and has correct columns
checks["train_processed.csv exists with comment_clean + comment_light"] = (
    os.path.exists(processed_path)
    and "comment_clean" in pd.read_csv(processed_path, nrows=1).columns
    and "comment_light" in pd.read_csv(processed_path, nrows=1).columns
)

# [2] No NaN in cleaned columns
_p = pd.read_csv(processed_path, usecols=["comment_clean", "comment_light"])
checks["No NaN in comment_clean or comment_light"] = (
    _p["comment_clean"].isna().sum() == 0
    and _p["comment_light"].isna().sum() == 0
)

# [3] class_weights.json has all 6 labels
with open(weights_path) as f:
    _w = json.load(f)
checks["class_weights.json has all 6 label keys"] = (
    set(_w.keys()) == set(LABEL_COLS)
)

# [4] tune_thresholds runs on dummy data
try:
    _d_proba = np.random.uniform(0, 1, (500, 6))
    _d_true  = (_d_proba > 0.7).astype(int)
    _bt = tune_thresholds(_d_true, _d_proba, LABEL_COLS)
    checks["tune_thresholds() runs on dummy data"] = (
        isinstance(_bt, dict) and len(_bt) == 6
    )
except Exception:
    checks["tune_thresholds() runs on dummy data"] = False

# [5] train_augmented.csv has more rows
_aug = pd.read_csv(augmented_path)
checks["train_augmented.csv has more rows than train_processed.csv"] = (
    len(_aug) > len(save_df)
)

# [6] threat count increased
checks["threat label count increased after augmentation"] = (
    int(_aug["threat"].sum()) > int(save_df["threat"].sum())
)

# [7] identity_hate count increased
checks["identity_hate label count increased after augmentation"] = (
    int(_aug["identity_hate"].sum()) > int(save_df["identity_hate"].sum())
)

# [8] preprocessor.py importable
try:
    spec2 = importlib.util.spec_from_file_location("preprocessor2", preprocessor_path)
    _m2   = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(_m2)
    checks["preprocessor.py exists and imports without errors"] = (
        hasattr(_m2, "clean_text_aggressive")
        and hasattr(_m2, "clean_text_light")
    )
except Exception:
    checks["preprocessor.py exists and imports without errors"] = False

print()
for desc, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    tick(desc, status)

all_pass = all(checks.values())
print()
if all_pass:
    print("  All checks passed! Phase 3 complete.")
else:
    n_fail = sum(1 for v in checks.values() if not v)
    print(f"  {n_fail} check(s) failed. Review output above.")

section("PHASE 3 COMPLETE")
print(f"""
  Outputs saved to: outputs/phase3/
    train_processed.csv    - {len(save_df):,} rows | comment_clean + comment_light added
    train_augmented.csv    - {len(df_augmented):,} rows | augmented threat + identity_hate
    class_weights.json     - per-label loss weights for Phase 4
    threshold_curves.png   - F1 vs threshold curves for all 6 labels
    preprocessor.py        - importable module for use in all future phases
""")
