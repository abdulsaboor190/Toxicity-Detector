# -*- coding: utf-8 -*-
"""
schemas.py — Pydantic request / response models for the toxicity API.
"""
from pydantic import BaseModel, Field
from typing import List


class AnalyzeRequest(BaseModel):
    """Incoming chat message to analyze."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The chat message to analyze for toxicity",
    )


class LabelScores(BaseModel):
    """Per-label probability scores (0.0 – 1.0)."""
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


class AnalyzeResponse(BaseModel):
    """Full analysis result returned to the client."""
    message: str                    # original message echoed back
    is_toxic: bool                  # True if ANY label exceeds its threshold
    severity: str                   # "clean" | "mild" | "toxic" | "severe"
    overall_score: float            # max score across all 6 labels (0.0–1.0)
    scores: LabelScores             # per-label probability scores
    flagged_categories: List[str]   # labels that exceeded their threshold
    processing_time_ms: float       # inference latency in milliseconds


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str
    model_loaded: bool
    device: str
    model_name: str
