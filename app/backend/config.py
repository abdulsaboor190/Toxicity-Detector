# -*- coding: utf-8 -*-
"""
config.py — Central configuration for the Chat Toxicity Detector API.

Environment variables (set in Render/Vercel dashboard):
    PORT               — auto-set by Render
    CORS_ALLOWED       — comma-separated origins (e.g. https://my-app.vercel.app)
"""
import os
import torch

# ── Paths (relative to project root) ─────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Best checkpoint from Phase 4
CHECKPOINT_FILENAME = "bert_epoch1_f10.4322.pt"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
THRESHOLDS_PATH = os.path.join(CHECKPOINT_DIR, "tuned_thresholds.json")

# Google Drive folder containing all model weights (auto-downloaded on first startup)
# Link: https://drive.google.com/drive/folders/1DmQdBI-r5MBX4aJTgmIJQAdSLhOpi41D
GDRIVE_MODELS_FOLDER_ID = "1DmQdBI-r5MBX4aJTgmIJQAdSLhOpi41D"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLS = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]

# ── Severity mapping (overall_score → human-readable level) ──────────────────
SEVERITY_LEVELS = {
    "clean":  (0.0, 0.2),
    "mild":   (0.2, 0.5),
    "toxic":  (0.5, 0.8),
    "severe": (0.8, 1.0),
}

# ── Server ───────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 8000))

# ── CORS ─────────────────────────────────────────────────────────────────────
# In production: set CORS_ALLOWED env var to your Vercel frontend URL
# e.g. CORS_ALLOWED=https://toxicity-detector.vercel.app
_default_origins = [
    "https://toxicity-detector-rho.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
    "*", # Wildcard catch-all to prevent strict CORS blocks
]

_env_origins = os.environ.get("CORS_ALLOWED", "")
if _env_origins:
    _default_origins.extend([o.strip() for o in _env_origins.split(",") if o.strip()])

CORS_ORIGINS = _default_origins
