"""Shared pytest fixtures for SemanticAPI tests."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from gensim.models import KeyedVectors

# ── Build a tiny in-memory KeyedVectors fixture ───────────────────────────────
VOCAB = ["king", "queen", "man", "woman", "breakfast", "lunch", "dinner", "cereal", "cat", "dog"]
DIM = 10
rng = np.random.default_rng(42)

_kv = KeyedVectors(vector_size=DIM)
_kv.add_vectors(VOCAB, rng.standard_normal((len(VOCAB), DIM)).astype(np.float32))


def _mock_get_model(lang: str) -> KeyedVectors:
    return _kv


@pytest.fixture()
def app():
    """Return a Flask test app with get_model patched to use the tiny fixture."""
    with patch("semanticapi.models.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.similarity.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.most_similar.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.analogy.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.odd_one_out.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.word_vector.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.sentence_similarity.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.routes.cluster.get_model", side_effect=_mock_get_model), \
         patch("semanticapi.models.preload_languages"):
        from semanticapi import create_app
        flask_app = create_app()
        flask_app.config["TESTING"] = True
        yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()
