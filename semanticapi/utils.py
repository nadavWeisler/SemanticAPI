"""Input normalisation and validation helpers."""
from __future__ import annotations

import re
import unicodedata

from flask import jsonify

from semanticapi.config import MAX_TOPN, MAX_WORD_LEN, MAX_WORDS_LIST_LEN


def normalise_word(word: str) -> str:
    """Normalise *word* for vocabulary lookup.

    Steps:
    1. NFC unicode normalisation.
    2. Strip leading/trailing whitespace.
    3. Lowercase.
    4. Replace internal whitespace runs with underscores (FastText convention
       for multi-word terms, e.g. ``new york`` → ``new_york``).
    """
    word = unicodedata.normalize("NFC", word)
    word = word.strip().lower()
    word = re.sub(r"\s+", "_", word)
    return word


def validate_word_length(word: str, field: str = "word"):
    """Return an error tuple if *word* exceeds MAX_WORD_LEN, else None."""
    if len(word) > MAX_WORD_LEN:
        return jsonify({"error": f"'{field}' exceeds maximum length of {MAX_WORD_LEN} characters"}), 400
    return None


def validate_topn(topn_raw, default: int = 10):
    """Parse and validate the *topn* query parameter.

    Returns ``(topn_int, None)`` on success or ``(None, error_tuple)`` on failure.
    """
    try:
        topn = int(topn_raw)
        if topn < 1:
            raise ValueError
    except (ValueError, TypeError):
        return None, (jsonify({"error": "'topn' must be a positive integer"}), 400)

    if topn > MAX_TOPN:
        topn = MAX_TOPN
    return topn, None


def validate_words_list_length(words_param: str):
    """Return an error tuple if *words_param* exceeds MAX_WORDS_LIST_LEN, else None."""
    if len(words_param) > MAX_WORDS_LIST_LEN:
        return jsonify({"error": f"'words' parameter exceeds maximum length of {MAX_WORDS_LIST_LEN} characters"}), 400
    return None
