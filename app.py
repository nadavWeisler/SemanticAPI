from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# ---- Configuration ----

# Language → Model path
LANG_MODELS = {
    'en': 'embeddings/cc.en.300.vec',
    'he': 'embeddings/cc.he.300.vec',
    'es': 'embeddings/cc.es.300.vec',
}

# ---- Routes ----

@app.route("/similarity", methods=["GET"])
def similarity():
    word1 = request.args.get("word1", "").lower()
    word2 = request.args.get("word2", "").lower()
    lang = request.args.get("lang", "en").lower()

    # Validate language
    if lang not in LANG_MODELS:
        return jsonify({"error": f"Language '{lang}' not supported"}), 400

    model = LANG_MODELS[lang]

    # Validate vocabulary
    if word1 not in model or word2 not in model:
        return jsonify({"error": "One or both words not in vocabulary"}), 400

    # Compute cosine similarity
    vec1 = model[word1].reshape(1, -1)
    vec2 = model[word2].reshape(1, -1)
    similarity_result = float(cosine_similarity(vec1, vec2)[0][0]) * 100

    return jsonify({"similarity": similarity_result})

# ---- Run App ----

if __name__ == "__main__":
    # Use PORT env var if set (e.g., for Render/Heroku), else default to 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
