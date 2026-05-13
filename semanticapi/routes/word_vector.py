from flask import Blueprint, request, jsonify

from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import normalise_word, validate_word_length

bp = Blueprint("word_vector", __name__)


@bp.route("/word-vector", methods=["GET"])
def word_vector():
    """
    Return the raw embedding vector for a word.
    ---
    tags:
      - Embeddings
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
    responses:
      200:
        description: Word embedding vector
        schema:
          type: object
          properties:
            word:
              type: string
            vector:
              type: array
              items:
                type: number
            dimensions:
              type: integer
            oov:
              type: boolean
      400:
        description: Validation error
    """
    word = normalise_word(request.args.get("word", ""))
    lang = request.args.get("lang", "en").lower().strip()

    if not word:
        return jsonify({"error": "'word' is required"}), 400

    err = validate_word_length(word)
    if err:
        return err

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    model = get_model(lang)

    try:
        vec, is_oov = get_word_vector(model, word)
    except KeyError:
        return jsonify({"error": f"Word '{word}' not in vocabulary"}), 400

    result = {
        "word": word,
        "vector": vec.tolist(),
        "dimensions": len(vec),
    }
    if is_oov:
        result["oov"] = True

    return jsonify(result)
