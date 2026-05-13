import threading
from flask import Blueprint, request, jsonify

from semanticapi.cache import _similarity_cache
from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import cosine_sim, normalise_word, validate_word_length

bp = Blueprint("similarity", __name__)

_cache_lock = threading.Lock()


@bp.route("/similarity", methods=["GET"])
def similarity():
    """
    Compute cosine similarity between two words.
    ---
    tags:
      - Similarity
    parameters:
      - name: word1
        in: query
        required: true
        type: string
      - name: word2
        in: query
        required: true
        type: string
      - name: lang
        in: query
        required: false
        type: string
        default: en
    responses:
      200:
        description: Similarity score (0-100)
        schema:
          type: object
          properties:
            similarity:
              type: number
            oov:
              type: boolean
      400:
        description: Validation error
    """
    word1 = normalise_word(request.args.get("word1", ""))
    word2 = normalise_word(request.args.get("word2", ""))
    lang = request.args.get("lang", "en").lower().strip()

    if not word1 or not word2:
        return jsonify({"error": "'word1' and 'word2' are required"}), 400

    err = validate_word_length(word1, "word1") or validate_word_length(word2, "word2")
    if err:
        return err

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    cache_key = ("similarity", word1, word2, lang)
    with _cache_lock:
        if cache_key in _similarity_cache:
            return jsonify(_similarity_cache[cache_key])

    model = get_model(lang)

    try:
        vec1, oov1 = get_word_vector(model, word1)
        vec2, oov2 = get_word_vector(model, word2)
    except KeyError as e:
        return jsonify({"error": f"Word '{e.args[0]}' not in vocabulary"}), 400

    score = cosine_sim(vec1, vec2)
    result = {"similarity": round(score, 2)}
    if oov1 or oov2:
        result["oov"] = True

    with _cache_lock:
        _similarity_cache[cache_key] = result

    return jsonify(result)


@bp.route("/similarity/batch", methods=["POST"])
def similarity_batch():
    """
    Compute similarity for multiple word pairs.
    ---
    tags:
      - Similarity
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - pairs
          properties:
            pairs:
              type: array
              items:
                type: object
                properties:
                  word1:
                    type: string
                  word2:
                    type: string
            lang:
              type: string
              default: en
    responses:
      200:
        description: Batch similarity results
      400:
        description: Validation error
    """
    data = request.get_json(silent=True)
    if not data or "pairs" not in data:
        return jsonify({"error": "'pairs' field is required in JSON body"}), 400

    lang = str(data.get("lang", "en")).lower().strip()
    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    pairs = data["pairs"]
    if not isinstance(pairs, list) or len(pairs) == 0:
        return jsonify({"error": "'pairs' must be a non-empty list"}), 400

    model = get_model(lang)
    results = []

    for pair in pairs:
        word1 = normalise_word(str(pair.get("word1", "")))
        word2 = normalise_word(str(pair.get("word2", "")))
        try:
            vec1, oov1 = get_word_vector(model, word1)
            vec2, oov2 = get_word_vector(model, word2)
            score = cosine_sim(vec1, vec2)
            entry = {"word1": word1, "word2": word2, "similarity": round(score, 2)}
            if oov1 or oov2:
                entry["oov"] = True
            results.append(entry)
        except KeyError as e:
            results.append({"word1": word1, "word2": word2, "error": f"Word '{e.args[0]}' not in vocabulary"})

    return jsonify({"results": results})
