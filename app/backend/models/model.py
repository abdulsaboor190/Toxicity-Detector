# -*- coding: utf-8 -*-
"""
model.py — ToxicityAnalyzer: loads model once, exposes .predict(text).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import json
import re
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import (
    MODEL_NAME, MAX_LENGTH, DEVICE, LABEL_COLS,
    CHECKPOINT_PATH, THRESHOLDS_PATH, SEVERITY_LEVELS,
    GDRIVE_MODELS_FOLDER_ID, CHECKPOINT_DIR,
)


# ── Inline light-cleaning (avoids spaCy dependency at serving time) ──────────
_RE_URL    = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML   = re.compile(r"<[^>]+>")
_RE_REPEAT = re.compile(r"(.)\1{2,}")


def clean_text_light(text: str) -> str:
    """Minimal cleaning preserving natural language for transformer input."""
    if not isinstance(text, str):
        return ""
    text = _RE_URL.sub("[URL]", text)
    text = _RE_HTML.sub("", text)
    text = _RE_REPEAT.sub(r"\1\1", text)
    return text.strip()


# ── Model Architecture (mirrors Phase 4) ────────────────────────────────────
class ToxicClassifier(nn.Module):
    """DistilBERT / BERT encoder + linear classification head."""

    def __init__(self, model_name: str, num_labels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


# ── Severity helper ─────────────────────────────────────────────────────────
def _severity(score: float) -> str:
    for level, (lo, hi) in SEVERITY_LEVELS.items():
        if lo <= score < hi:
            return level
    return "severe"


# ── Main Analyzer ───────────────────────────────────────────────────────────
class ToxicityAnalyzer:
    """
    Singleton-style analyzer.

    Usage:
        analyzer = ToxicityAnalyzer()
        analyzer.load()            # call once at startup
        result = analyzer.predict("some text")
    """

    def __init__(self):
        self.tokenizer  = None
        self.model      = None
        self.thresholds = None
        self.is_loaded  = False

    # ------------------------------------------------------------------ load
    def load(self):
        """Load tokenizer, model weights, and per-label thresholds."""
        print(f"  [model] Loading tokenizer: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print(f"  [model] Building architecture: {MODEL_NAME}")
        self.model = ToxicClassifier(MODEL_NAME, num_labels=len(LABEL_COLS))

        # Load fine-tuned weights — auto-download from Google Drive if missing
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"  [model] Checkpoint not found locally. Downloading from Google Drive...")
            try:
                import gdown
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                url = f"https://drive.google.com/drive/folders/{GDRIVE_MODELS_FOLDER_ID}"
                gdown.download_folder(url, output=CHECKPOINT_DIR, quiet=False, use_cookies=False)
                print(f"  [model] Download complete: {CHECKPOINT_DIR}")
            except Exception as e:
                print(f"  [model] ERROR — Failed to download checkpoint: {e}")
                print(f"  [model] WARNING — Using base weights (no fine-tuning)")

        if os.path.exists(CHECKPOINT_PATH):
            print(f"  [model] Loading checkpoint: {os.path.basename(CHECKPOINT_PATH)} (mmap enabled)")
            # mmap=True drops peak RAM usage from 1.5GB+ down to nearly 0!
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False, mmap=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            print("  [model] Checkpoint loaded.")
        else:
            print(f"  [model] WARNING — checkpoint not found, using base weights")

        self.model = self.model.to(DEVICE)
        self.model.eval()
        torch.set_grad_enabled(False)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [model] Parameters: {n_params:,}")
        print(f"  [model] Device: {DEVICE}")

        # Thresholds
        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH) as f:
                self.thresholds = json.load(f)
            print(f"  [model] Thresholds loaded: {self.thresholds}")
        else:
            # Fallback defaults
            self.thresholds = {lbl: 0.5 for lbl in LABEL_COLS}
            print("  [model] WARNING — thresholds file not found, using 0.5 defaults")

        self.is_loaded = True
        print("  [model] Ready for inference.")

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Run inference on a single message.

        Returns dict matching AnalyzeResponse fields (minus processing_time_ms).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call analyzer.load() first.")

        cleaned = clean_text_light(text)

        # Edge case: empty after cleaning
        if not cleaned.strip():
            return {
                "message":            text,
                "is_toxic":           False,
                "severity":           "clean",
                "overall_score":      0.0,
                "scores":             {lbl: 0.0 for lbl in LABEL_COLS},
                "flagged_categories": [],
                "processing_time_ms": 0.0,
            }

        # Tokenize
        enc = self.tokenizer(
            cleaned,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        # Forward pass
        logits = self.model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

        # Per-label results
        scores = {lbl: float(round(probs[i], 4)) for i, lbl in enumerate(LABEL_COLS)}
        flagged = [
            lbl for i, lbl in enumerate(LABEL_COLS)
            if probs[i] >= self.thresholds[lbl]
        ]
        overall_score = float(probs.max())

        return {
            "message":            text,
            "is_toxic":           len(flagged) > 0,
            "severity":           _severity(overall_score),
            "overall_score":      round(overall_score, 4),
            "scores":             scores,
            "flagged_categories": flagged,
            "processing_time_ms": 0.0,          # filled by the route handler
        }


# ── Global instance ─────────────────────────────────────────────────────────
analyzer = ToxicityAnalyzer()
