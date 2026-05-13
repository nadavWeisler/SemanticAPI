import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import normalise_word

bp = Blueprint("sentence_similarity", __name__)

# Common English stop words to skip when averaging (optional)
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could",
}


def _sentence_vector(model, text: str, skip_stopwords: bool = False) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Compute the mean vector of words in *text*.

    Returns (mean_vector_or_None, known_words, oov_words).
    """
    tokens = [normalise_word(t) for t in text.split() if t.strip()]
    if skip_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    vecs = []
    oov_words = []
    known_words = []

    for token in tokens:
        try:
            vec, is_oov = get_word_vector(model, token)
            vecs.append(vec)
            if is_oov:
                oov_words.append(token)
            else:
                known_words.append(token)
        except KeyError:
            pass  # Skip truly unknown tokens silently

    if not vecs:
        return None, known_words, oov_words

    return np.mean(vecs, axis=0), known_words, oov_words


@bp.route("/sentence-similarity", methods=["GET"])
def sentence_similarity():
    """
    Compute semantic similarity between two phrases/sentences.
    ---
    tags:
      - Similarity
    parameters:
      - name: text1
        in: query
        required: true
        type: string
      - name: text2
        in: query
        required: true
        type: string
      - name: lang
        in: query
        required: false
        type: string
        default: en
      - name: skip_stopwords
        in: query
        required: false
        type: boolean
        default: false
    responses:
      200:
        description: Sentence similarity score (0-100)
      400:
        description: Validation error
    """
    text1 = request.args.get("text1", "").strip()
    text2 = request.args.get("text2", "").strip()
    lang = request.args.get("lang", "en").lower().strip()
    skip_stopwords = request.args.get("skip_stopwords", "false").lower() in ("1", "true", "yes")

    if not text1 or not text2:
        return jsonify({"error": "'text1' and 'text2' are required"}), 400

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    model = get_model(lang)

    vec1, known1, oov1 = _sentence_vector(model, text1, skip_stopwords)
    vec2, known2, oov2 = _sentence_vector(model, text2, skip_stopwords)

    if vec1 is None or vec2 is None:
        return jsonify({"error": "Could not compute vectors — no known words found in one or both texts"}), 400

    score = float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]) * 100

    result = {"similarity": round(score, 2)}
    if oov1 or oov2:
        result["oov_words"] = list(set(oov1 + oov2))

    return jsonify(result)
