import numpy as np
from flask import Blueprint, request, jsonify

from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import normalise_word, validate_words_list_length

bp = Blueprint("odd_one_out", __name__)


@bp.route("/odd-one-out", methods=["GET"])
def odd_one_out():
    """
    Identify the word that does not belong among a list of words.
    ---
    tags:
      - Analogy
    parameters:
      - name: words
        in: query
        required: true
        type: string
        description: Comma-separated list of words (min 3)
      - name: lang
        in: query
        required: false
        type: string
        default: en
    responses:
      200:
        description: Odd-one-out result with full ranking
      400:
        description: Validation error
    """
    words_param = request.args.get("words", "")
    lang = request.args.get("lang", "en").lower().strip()

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    err = validate_words_list_length(words_param)
    if err:
        return err

    words = [normalise_word(w) for w in words_param.split(",") if w.strip()]

    if len(words) < 3:
        return jsonify({"error": "At least 3 words are required"}), 400

    model = get_model(lang)

    vectors = {}
    oov_words = []
    for w in words:
        try:
            vec, is_oov = get_word_vector(model, w)
            vectors[w] = vec
            if is_oov:
                oov_words.append(w)
        except KeyError:
            return jsonify({"error": f"Word '{w}' not in vocabulary"}), 400

    # Compute centroid
    all_vecs = np.stack(list(vectors.values()))
    centroid = all_vecs.mean(axis=0)

    # Rank each word by distance from centroid (higher = more odd)
    ranking = []
    for w in words:
        vec = vectors[w]
        dist = float(np.linalg.norm(vec - centroid))
        ranking.append({"word": w, "oddness_score": round(dist, 6)})

    ranking.sort(key=lambda x: x["oddness_score"], reverse=True)
    odd_word = ranking[0]["word"]

    result = {
        "odd_one_out": odd_word,
        "words": words,
        "ranking": ranking,
    }
    if oov_words:
        result["oov"] = oov_words

    return jsonify(result)
