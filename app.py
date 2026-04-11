from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import os
import struct

app = Flask(__name__)

# ---- Configuration ----

EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", "embeddings")

# Language → Local model path
LANG_MODEL_PATHS = {
    'en': os.path.join(EMBEDDINGS_DIR, 'cc.en.300.vec'),
    'he': os.path.join(EMBEDDINGS_DIR, 'cc.he.300.vec'),
    'es': os.path.join(EMBEDDINGS_DIR, 'cc.es.300.vec'),
}

# Language → Hugging Face Hub model config (repo_id, filename_in_repo).
# Override individual entries via environment variables:
#   HF_REPO_<LANG> and HF_FILE_<LANG>   (e.g. HF_REPO_EN, HF_FILE_EN)
LANG_HF_MODELS = {
    'en': (
        os.environ.get('HF_REPO_EN', 'facebook/fasttext-en-vectors'),
        os.environ.get('HF_FILE_EN', 'model.bin'),
    ),
    'he': (
        os.environ.get('HF_REPO_HE', 'facebook/fasttext-he-vectors'),
        os.environ.get('HF_FILE_HE', 'model.bin'),
    ),
    'es': (
        os.environ.get('HF_REPO_ES', 'facebook/fasttext-es-vectors'),
        os.environ.get('HF_FILE_ES', 'model.bin'),
    ),
}

# Loaded models cache: lang → KeyedVectors
_loaded_models = {}


def _load_keyed_vectors(path):
    """Load a KeyedVectors model from *path*, auto-detecting the file format."""
    if path.endswith('.bin'):
        try:
            from gensim.models.fasttext import load_facebook_vectors
            return load_facebook_vectors(path)
        except (ValueError, EOFError, struct.error):
            return KeyedVectors.load_word2vec_format(path, binary=True)
    return KeyedVectors.load_word2vec_format(path)


def download_model_from_hf(lang):
    """Download the model for *lang* from Hugging Face Hub.

    The file is saved into *EMBEDDINGS_DIR* and ``LANG_MODEL_PATHS[lang]``
    is updated to point at the downloaded path so that subsequent calls to
    :func:`get_model` use the correct location.

    Set the ``HF_TOKEN`` environment variable to authenticate with private
    or gated repositories.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic model downloads. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    repo_id, filename = LANG_HF_MODELS[lang]
    token = os.environ.get('HF_TOKEN')

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print(f"[SemanticAPI] Downloading '{lang}' model from {repo_id}/{filename} …")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=EMBEDDINGS_DIR,
    )
    print(f"[SemanticAPI] Model saved to {downloaded_path}")

    LANG_MODEL_PATHS[lang] = downloaded_path
    return downloaded_path


def get_model(lang):
    """Return the KeyedVectors model for *lang*, loading it on first use.

    If the configured local model file does not exist and ``huggingface_hub``
    is installed, the model is automatically downloaded from Hugging Face Hub
    before loading.
    """
    if lang not in _loaded_models:
        path = LANG_MODEL_PATHS[lang]

        if not os.path.exists(path):
            path = download_model_from_hf(lang)

        _loaded_models[lang] = _load_keyed_vectors(path)
    return _loaded_models[lang]


def validate_lang(lang):
    """Return an error response tuple if *lang* is unsupported, else None."""
    if lang not in LANG_MODEL_PATHS:
        return jsonify({"error": f"Language '{lang}' not supported"}), 400
    return None


# ---- Routes ----

@app.route("/similarity", methods=["GET"])
def similarity():
    word1 = request.args.get("word1", "").lower()
    word2 = request.args.get("word2", "").lower()
    lang = request.args.get("lang", "en").lower()

    err = validate_lang(lang)
    if err:
        return err

    model = get_model(lang)

    # Validate vocabulary
    if word1 not in model or word2 not in model:
        return jsonify({"error": "One or both words not in vocabulary"}), 400

    # Compute cosine similarity
    vec1 = model[word1].reshape(1, -1)
    vec2 = model[word2].reshape(1, -1)
    similarity_result = float(cosine_similarity(vec1, vec2)[0][0]) * 100

    return jsonify({"similarity": similarity_result})


@app.route("/most-similar", methods=["GET"])
def most_similar():
    """Return the *topn* words most similar to *word*."""
    word = request.args.get("word", "").lower()
    lang = request.args.get("lang", "en").lower()
    topn = request.args.get("topn", 10)

    err = validate_lang(lang)
    if err:
        return err

    try:
        topn = int(topn)
        if topn < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "'topn' must be a positive integer"}), 400

    model = get_model(lang)

    if word not in model:
        return jsonify({"error": f"Word '{word}' not in vocabulary"}), 400

    results = model.most_similar(word, topn=topn)
    similar_words = [{"word": w, "similarity": round(score * 100, 2)} for w, score in results]

    return jsonify({"word": word, "similar": similar_words})


@app.route("/analogy", methods=["GET"])
def analogy():
    """Solve word analogies: *positive1* - *negative* + *positive2* = ?

    Example: king - man + woman = queen
      word1=king&word2=woman&negative=man
    """
    word1 = request.args.get("word1", "").lower()
    word2 = request.args.get("word2", "").lower()
    negative = request.args.get("negative", "").lower()
    lang = request.args.get("lang", "en").lower()
    topn = request.args.get("topn", 5)

    err = validate_lang(lang)
    if err:
        return err

    if not word1 or not word2 or not negative:
        return jsonify({"error": "'word1', 'word2', and 'negative' are required"}), 400

    try:
        topn = int(topn)
        if topn < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "'topn' must be a positive integer"}), 400

    model = get_model(lang)

    missing = [w for w in (word1, word2, negative) if w not in model]
    if missing:
        return jsonify({"error": f"Words not in vocabulary: {missing}"}), 400

    results = model.most_similar(positive=[word1, word2], negative=[negative], topn=topn)
    analogy_results = [{"word": w, "similarity": round(score * 100, 2)} for w, score in results]

    return jsonify({"analogy": analogy_results})


@app.route("/odd-one-out", methods=["GET"])
def odd_one_out():
    """Identify the word that does not belong among a list of words.

    Pass words as a comma-separated *words* query parameter.
    Example: ?words=breakfast,cereal,lunch,dinner
    """
    words_param = request.args.get("words", "")
    lang = request.args.get("lang", "en").lower()

    err = validate_lang(lang)
    if err:
        return err

    words = [w.strip().lower() for w in words_param.split(",") if w.strip()]

    if len(words) < 3:
        return jsonify({"error": "At least 3 words are required"}), 400

    model = get_model(lang)

    missing = [w for w in words if w not in model]
    if missing:
        return jsonify({"error": f"Words not in vocabulary: {missing}"}), 400

    odd_word = model.doesnt_match(words)

    return jsonify({"odd_one_out": odd_word, "words": words})


# ---- Run App ----

if __name__ == "__main__":
    # Use PORT env var if set (e.g., for Render/Heroku), else default to 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
