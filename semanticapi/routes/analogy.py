from flask import Blueprint, request, jsonify

from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import normalise_word, validate_topn, validate_word_length

bp = Blueprint("analogy", __name__)


@bp.route("/analogy", methods=["GET"])
def analogy():
    """
    Solve word-vector analogies: word1 − negative + word2 = ?
    ---
    tags:
      - Analogy
    parameters:
      - name: word1
        in: query
        required: true
        type: string
      - name: word2
        in: query
        required: true
        type: string
      - name: negative
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
        default: 5
    responses:
      200:
        description: Analogy results
      400:
        description: Validation error
    """
    word1 = normalise_word(request.args.get("word1", ""))
    word2 = normalise_word(request.args.get("word2", ""))
    negative = normalise_word(request.args.get("negative", ""))
    lang = request.args.get("lang", "en").lower().strip()
    topn_raw = request.args.get("topn", 5)

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    if not word1 or not word2 or not negative:
        return jsonify({"error": "'word1', 'word2', and 'negative' are required"}), 400

    for field, val in [("word1", word1), ("word2", word2), ("negative", negative)]:
        err = validate_word_length(val, field)
        if err:
            return err

    topn, err = validate_topn(topn_raw, default=5)
    if err:
        return err

    model = get_model(lang)

    vecs = {}
    for name, w in [("word1", word1), ("word2", word2), ("negative", negative)]:
        try:
            vec, _ = get_word_vector(model, w)
            vecs[name] = vec
        except KeyError:
            return jsonify({"error": f"Word '{w}' not in vocabulary"}), 400

    results = model.most_similar(
        positive=[vecs["word1"], vecs["word2"]],
        negative=[vecs["negative"]],
        topn=topn,
    )
    analogy_results = [{"word": w, "similarity": round(score * 100, 2)} for w, score in results]

    return jsonify({"analogy": analogy_results})
