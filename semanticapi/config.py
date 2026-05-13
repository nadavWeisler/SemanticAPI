"""Centralised configuration — all values read from environment variables."""
import os

# Embeddings storage
EMBEDDINGS_DIR: str = os.environ.get("EMBEDDINGS_DIR", "embeddings")

# Languages to eagerly load at startup (comma-separated, e.g. "en,es")
_preload_raw = os.environ.get("PRELOAD_LANGS", "")
PRELOAD_LANGS: list[str] = [x.strip() for x in _preload_raw.split(",") if x.strip()]

# Input validation limits
MAX_TOPN: int = int(os.environ.get("MAX_TOPN", "200"))
MAX_WORD_LEN: int = int(os.environ.get("MAX_WORD_LEN", "200"))
MAX_WORDS_LIST_LEN: int = int(os.environ.get("MAX_WORDS_LIST_LEN", "2000"))

# Rate limiting (Flask-Limiter format, e.g. "60 per minute")
RATE_LIMIT_DEFAULT: str = os.environ.get("RATE_LIMIT_DEFAULT", "60 per minute")

# Response cache TTL in seconds
CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "3600"))
CACHE_MAXSIZE: int = int(os.environ.get("CACHE_MAXSIZE", "1024"))

# HuggingFace Hub settings
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
