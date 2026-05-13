"""Model loading, caching, and OOV vector helpers."""
from __future__ import annotations

import os
import struct
import threading
from typing import TYPE_CHECKING

import numpy as np
from gensim.models import KeyedVectors

from semanticapi.config import EMBEDDINGS_DIR, HF_TOKEN, PRELOAD_LANGS

if TYPE_CHECKING:
    pass

# ── Internal state ────────────────────────────────────────────────────────────
_models: dict[str, KeyedVectors] = {}
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()  # protects _locks dict itself


def _get_lock(lang: str) -> threading.Lock:
    with _locks_lock:
        if lang not in _locks:
            _locks[lang] = threading.Lock()
        return _locks[lang]


# ── HuggingFace model resolution ──────────────────────────────────────────────

def _hf_config(lang: str) -> tuple[str, str]:
    """Return (repo_id, filename) for *lang*, honoring env-var overrides."""
    lang_up = lang.upper()
    repo = os.environ.get(f"HF_REPO_{lang_up}", f"facebook/fasttext-{lang}-vectors")
    file = os.environ.get(f"HF_FILE_{lang_up}", "model.bin")
    return repo, file


def _local_path(lang: str) -> str:
    """Return the expected local path for *lang*'s model file."""
    repo, filename = _hf_config(lang)
    # Use the filename component as the local name
    return os.path.join(EMBEDDINGS_DIR, filename if filename != "model.bin" else f"model_{lang}.bin")


def _download(lang: str) -> str:
    """Download model for *lang* from HuggingFace Hub and return local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic model downloads. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    repo_id, filename = _hf_config(lang)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print(f"[SemanticAPI] Downloading '{lang}' model from {repo_id}/{filename} …")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=HF_TOKEN,
        local_dir=EMBEDDINGS_DIR,
    )
    print(f"[SemanticAPI] Model saved to {path}")
    return path


def _load_keyed_vectors(path: str) -> KeyedVectors:
    """Load KeyedVectors from *path*, auto-detecting format."""
    if path.endswith(".bin"):
        try:
            from gensim.models.fasttext import load_facebook_vectors
            return load_facebook_vectors(path)
        except (ValueError, EOFError, struct.error):
            return KeyedVectors.load_word2vec_format(path, binary=True)
    return KeyedVectors.load_word2vec_format(path)


# ── Public API ────────────────────────────────────────────────────────────────

def is_valid_lang(lang: str) -> bool:
    """Return True if *lang* looks like a valid ISO 639-1/2 language code."""
    return lang.isalpha() and 2 <= len(lang) <= 3


def get_model(lang: str) -> KeyedVectors:
    """Return the KeyedVectors model for *lang*, loading it on first use (thread-safe)."""
    if lang in _models:
        return _models[lang]

    lock = _get_lock(lang)
    with lock:
        # Double-checked locking
        if lang in _models:
            return _models[lang]

        # Find or download the model file
        path = _local_path(lang)
        if not os.path.exists(path):
            path = _download(lang)

        _models[lang] = _load_keyed_vectors(path)
    return _models[lang]


def supports_oov(model: KeyedVectors) -> bool:
    """Return True if *model* can produce vectors for out-of-vocabulary words."""
    try:
        from gensim.models.fasttext import FastTextKeyedVectors
        return isinstance(model, FastTextKeyedVectors)
    except ImportError:
        return False


def get_word_vector(model: KeyedVectors, word: str) -> tuple[np.ndarray, bool]:
    """Return *(vector, is_oov)* for *word*.

    Raises ``KeyError`` if the word is not in vocabulary and the model does
    not support subword (OOV) vectors.
    """
    if word in model:
        return model[word], False
    if supports_oov(model):
        vec = model.get_vector(word, norm=False)
        return vec, True
    raise KeyError(word)


def loaded_languages() -> list[str]:
    """Return list of currently-loaded language codes."""
    return list(_models.keys())


def preload_languages() -> None:
    """Eagerly load languages listed in PRELOAD_LANGS config."""
    for lang in PRELOAD_LANGS:
        if is_valid_lang(lang):
            try:
                get_model(lang)
                print(f"[SemanticAPI] Preloaded model for '{lang}'")
            except Exception as exc:  # noqa: BLE001
                print(f"[SemanticAPI] Warning: could not preload '{lang}': {exc}")
