import threading
from flask import Blueprint, request, jsonify

from semanticapi.cache import _most_similar_cache
from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import normalise_word, validate_topn, validate_word_length

bp = Blueprint("most_similar", __name__)

_cache_lock = threading.Lock()


@bp.route("/most-similar", methods=["GET"])
def most_similar():
    """
    Return the N words most similar to a given word.
    ---
    tags:
      - Similarity
    parameters:
      - name: word
        in: query
        required: true
        type: string
      - name: lang
        in: query
        required: false
        type: string
        default: en
      - name: topn
        in: query
        required: false
        type: integer
        default: 10
    responses:
      200:
        description: Most similar words
      400:
        description: Validation error
    """
    word = normalise_word(request.args.get("word", ""))
    lang = request.args.get("lang", "en").lower().strip()
    topn_raw = request.args.get("topn", 10)

    if not word:
        return jsonify({"error": "'word' is required"}), 400

    err = validate_word_length(word)
    if err:
        return err

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    topn, err = validate_topn(topn_raw, default=10)
    if err:
        return err

    cache_key = ("most_similar", word, lang, topn)
    cached = None
    with _cache_lock:
        if cache_key in _most_similar_cache:
            cached = dict(_most_similar_cache[cache_key])

    if cached is not None:
        return jsonify(cached)

    model = get_model(lang)

    try:
        vec, is_oov = get_word_vector(model, word)
    except KeyError:
        return jsonify({"error": f"Word '{word}' not in vocabulary"}), 400

    # most_similar accepts a vector directly
    results = model.similar_by_vector(vec, topn=topn)
    similar_words = [{"word": w, "similarity": round(score * 100, 2)} for w, score in results]

    result = {"word": word, "similar": similar_words}
    if is_oov:
        result["oov"] = True

    with _cache_lock:
        _most_similar_cache[cache_key] = result

    return jsonify(result)


@bp.route("/most-similar/batch", methods=["POST"])
def most_similar_batch():
    """
    Return most-similar lists for multiple words in one request.
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
            - words
          properties:
            words:
              type: array
              items:
                type: string
            lang:
              type: string
              default: en
            topn:
              type: integer
              default: 10
    responses:
      200:
        description: Batch most-similar results
      400:
        description: Validation error
    """
    data = request.get_json(silent=True)
    if not data or "words" not in data:
        return jsonify({"error": "'words' field is required in JSON body"}), 400

    lang = str(data.get("lang", "en")).lower().strip()
    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    topn, err = validate_topn(data.get("topn", 10))
    if err:
        return err

    words = data["words"]
    if not isinstance(words, list) or len(words) == 0:
        return jsonify({"error": "'words' must be a non-empty list"}), 400

    model = get_model(lang)
    results = []

    for raw_word in words:
        word = normalise_word(str(raw_word))
        try:
            vec, is_oov = get_word_vector(model, word)
            similar = model.similar_by_vector(vec, topn=topn)
            entry = {
                "word": word,
                "similar": [{"word": w, "similarity": round(s * 100, 2)} for w, s in similar],
            }
            if is_oov:
                entry["oov"] = True
            results.append(entry)
        except KeyError:
            results.append({"word": word, "error": f"Word '{word}' not in vocabulary"})

    return jsonify({"results": results})
