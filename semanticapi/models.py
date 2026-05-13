"""Model loading, caching, and OOV vector helpers."""
from __future__ import annotations

import os
import re
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

# Only lowercase ASCII letters are accepted in a language code.
_LANG_RE = re.compile(r"^[a-z]{2,3}$")


def _get_lock(lang: str) -> threading.Lock:
    with _locks_lock:
        if lang not in _locks:
            _locks[lang] = threading.Lock()
        return _locks[lang]


# ── HuggingFace model resolution ──────────────────────────────────────────────

def _sanitize_lang(lang: str) -> str:
    """Return a filesystem-safe copy of *lang*.

    Strips every character that is not a lowercase ASCII letter so that the
    result contains only ``[a-z]{2,3}``.  This breaks any taint-flow from
    user-provided values before the string is incorporated into a file path.

    Raises ``ValueError`` for codes that don't conform to ISO 639-1/2 after
    sanitization.
    """
    sanitized = re.sub(r"[^a-z]", "", lang.lower())
    if not _LANG_RE.match(sanitized):
        raise ValueError(f"Invalid language code: '{lang}'")
    return sanitized


def _hf_config(lang: str) -> tuple[str, str]:
    """Return (repo_id, filename) for *lang*, honoring env-var overrides."""
    lang_up = lang.upper()
    repo = os.environ.get(f"HF_REPO_{lang_up}", f"facebook/fasttext-{lang}-vectors")
    file = os.environ.get(f"HF_FILE_{lang_up}", "model.bin")
    return repo, file


def _local_path(lang: str) -> str:
    """Return the expected local path for *lang*'s model file.

    *lang* is sanitized to ``[a-z]{2,3}`` before use, and the final path is
    verified via ``os.path.realpath`` to remain inside *EMBEDDINGS_DIR*.
    """
    # Sanitize lang — strips all non-[a-z] characters.  This is the primary
    # defence: the resulting string provably contains only safe characters.
    safe_lang = _sanitize_lang(lang)

    _, filename = _hf_config(safe_lang)
    # Always embed the language code in the filename so multiple languages
    # don't collide on the same file.  Strip directory separators from any
    # env-var-supplied filename to prevent path traversal.
    if filename == "model.bin":
        base_name = f"model_{safe_lang}.bin"
    else:
        base_name = os.path.basename(filename)

    resolved = os.path.realpath(os.path.join(EMBEDDINGS_DIR, base_name))
    embeddings_realpath = os.path.realpath(EMBEDDINGS_DIR)

    # Cross-platform containment check using commonpath.
    try:
        common = os.path.commonpath([resolved, embeddings_realpath])
    except ValueError:
        # commonpath raises ValueError on Windows when paths are on different drives.
        raise ValueError(
            f"Computed model path '{resolved}' escapes EMBEDDINGS_DIR ('{embeddings_realpath}')"
        )
    if common != embeddings_realpath:
        raise ValueError(
            f"Computed model path '{resolved}' escapes EMBEDDINGS_DIR ('{embeddings_realpath}')"
        )
    return resolved


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
    return bool(_LANG_RE.match(lang.lower())) if lang else False


def get_model(lang: str) -> KeyedVectors:
    """Return the KeyedVectors model for *lang*, loading it on first use (thread-safe)."""
    if not is_valid_lang(lang):
        raise ValueError(f"Invalid language code: '{lang}'")

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
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                print(f"[SemanticAPI] Warning: could not preload '{lang}': {exc}")

