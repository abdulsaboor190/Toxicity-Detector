# -*- coding: utf-8 -*-
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
