# -*- coding: utf-8 -*-
"""
main.py — FastAPI application for the Chat Toxicity Detector.

Run locally:
    cd app/backend
    uvicorn main:app --reload --port 8000
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import time
import threading
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from config import CORS_ORIGINS, DEVICE, MODEL_NAME
from models.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from models.model import analyzer


# ── Session stats store (in-memory) ─────────────────────────────────────────
_stats_lock = threading.Lock()
_stats = {
    "total_analyzed": 0,
    "total_toxic": 0,
    "severity_counts": {"clean": 0, "mild": 0, "toxic": 0, "severe": 0},
    "label_counts": {"toxic": 0, "severe_toxic": 0, "obscene": 0,
                     "threat": 0, "insult": 0, "identity_hate": 0},
    "label_score_sums": {"toxic": 0.0, "severe_toxic": 0.0, "obscene": 0.0,
                         "threat": 0.0, "insult": 0.0, "identity_hate": 0.0},
    "processing_times": [],
    "history": [],          # last 100 results (timestamp, severity, overall_score)
    "session_start": time.time(),
}


def _record_stats(result: dict):
    with _stats_lock:
        _stats["total_analyzed"] += 1
        if result["is_toxic"]:
            _stats["total_toxic"] += 1
        sev = result["severity"]
        _stats["severity_counts"][sev] = _stats["severity_counts"].get(sev, 0) + 1
        for lbl, score in result["scores"].items():
            if isinstance(result["scores"], dict):
                s = score
            else:
                s = getattr(result["scores"], lbl, 0.0)
            _stats["label_score_sums"][lbl] = _stats["label_score_sums"].get(lbl, 0.0) + s
        for cat in result["flagged_categories"]:
            _stats["label_counts"][cat] = _stats["label_counts"].get(cat, 0) + 1
        _stats["processing_times"].append(result["processing_time_ms"])
        if len(_stats["processing_times"]) > 200:
            _stats["processing_times"] = _stats["processing_times"][-200:]
        _stats["history"].append({
            "t": round(time.time() - _stats["session_start"], 1),
            "severity": sev,
            "score": result["overall_score"],
        })
        if len(_stats["history"]) > 100:
            _stats["history"] = _stats["history"][-100:]


# -- Lifespan (load model on startup) ----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts."""
    print("\n  ================================================")
    print("    Chat Toxicity Detector -- Starting up...")
    print("  ================================================\n")
    analyzer.load()
    yield
    print("\n  Server shutting down.")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chat Toxicity Detector API",
    description=(
        "Real-time multi-label toxicity classification for chat messages. "
        "Powered by a fine-tuned DistilBERT model trained on the "
        "Jigsaw Toxic Comment Classification dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verify model is loaded and ready."""
    return HealthResponse(
        status="ok" if analyzer.is_loaded else "loading",
        model_loaded=analyzer.is_loaded,
        device=DEVICE,
        model_name=MODEL_NAME,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a single chat message for toxicity.

    Returns per-label scores, overall severity, and flagged categories.
    """
    if not analyzer.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading")

    start = time.perf_counter()

    try:
        result = analyzer.predict(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 1)
    _record_stats(result)
    return AnalyzeResponse(**result)


@app.get("/stats")
async def get_stats():
    """
    Return session-level analytics: counts, averages, history.
    """
    with _stats_lock:
        n = _stats["total_analyzed"]
        avg_scores = {
            lbl: round(_stats["label_score_sums"][lbl] / n, 4) if n > 0 else 0.0
            for lbl in _stats["label_score_sums"]
        }
        times = _stats["processing_times"]
        avg_latency = round(sum(times) / len(times), 1) if times else 0.0
        return {
            "total_analyzed": n,
            "total_toxic": _stats["total_toxic"],
            "toxicity_rate": round(_stats["total_toxic"] / n, 4) if n > 0 else 0.0,
            "severity_counts": dict(_stats["severity_counts"]),
            "label_counts": dict(_stats["label_counts"]),
            "avg_label_scores": avg_scores,
            "avg_latency_ms": avg_latency,
            "history": list(_stats["history"]),
            "session_uptime_s": round(time.time() - _stats["session_start"], 0),
        }


@app.post("/stats/reset")
async def reset_stats():
    """Reset all session statistics."""
    with _stats_lock:
        _stats["total_analyzed"] = 0
        _stats["total_toxic"] = 0
        _stats["severity_counts"] = {"clean": 0, "mild": 0, "toxic": 0, "severe": 0}
        _stats["label_counts"] = {lbl: 0 for lbl in _stats["label_counts"]}
        _stats["label_score_sums"] = {lbl: 0.0 for lbl in _stats["label_score_sums"]}
        _stats["processing_times"] = []
        _stats["history"] = []
        _stats["session_start"] = time.time()
    return {"status": "reset"}


@app.post("/analyze/batch")
async def analyze_batch(messages: List[str]):
    """
    Analyze up to 50 messages in one call. Useful for testing.
    """
    if not analyzer.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if len(messages) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 messages per batch")

    results = []
    for msg in messages:
        t0 = time.perf_counter()
        r = analyzer.predict(msg)
        r["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results.append(r)
    return results
