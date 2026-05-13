import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.cluster import KMeans

from semanticapi.models import get_model, get_word_vector, is_valid_lang
from semanticapi.utils import normalise_word, validate_words_list_length

bp = Blueprint("cluster", __name__)


@bp.route("/cluster", methods=["GET"])
def cluster():
    """
    Cluster a list of words into k groups using k-means on their embeddings.
    ---
    tags:
      - Embeddings
    parameters:
      - name: words
        in: query
        required: true
        type: string
        description: Comma-separated list of words (min 2)
      - name: k
        in: query
        required: false
        type: integer
        default: 3
        description: Number of clusters
      - name: lang
        in: query
        required: false
        type: string
        default: en
    responses:
      200:
        description: Cluster assignments
      400:
        description: Validation error
    """
    words_param = request.args.get("words", "")
    lang = request.args.get("lang", "en").lower().strip()
    k_raw = request.args.get("k", 3)

    if not is_valid_lang(lang):
        return jsonify({"error": f"Language '{lang}' is not a valid ISO language code"}), 400

    err = validate_words_list_length(words_param)
    if err:
        return err

    words = [normalise_word(w) for w in words_param.split(",") if w.strip()]

    if len(words) < 2:
        return jsonify({"error": "At least 2 words are required"}), 400

    try:
        k = int(k_raw)
        if k < 1:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "'k' must be a positive integer"}), 400

    if k > len(words):
        return jsonify({"error": f"'k' ({k}) cannot exceed the number of words ({len(words)})"}), 400

    model = get_model(lang)

    vecs = []
    oov_words = []
    valid_words = []
    for w in words:
        try:
            vec, is_oov = get_word_vector(model, w)
            vecs.append(vec)
            valid_words.append(w)
            if is_oov:
                oov_words.append(w)
        except KeyError:
            return jsonify({"error": f"Word '{w}' not in vocabulary"}), 400

    X = np.stack(vecs)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    clusters: dict[int, list[str]] = {i: [] for i in range(k)}
    for word, label in zip(valid_words, labels):
        clusters[int(label)].append(word)

    result = {"k": k, "clusters": list(clusters.values())}
    if oov_words:
        result["oov"] = oov_words

    return jsonify(result)
