# SemanticAPI

A REST API for semantic word similarity, analogies, clustering and more — powered by [FastText](https://fasttext.cc/) embeddings via [Gensim](https://radimrehurek.com/gensim/).

## Features

- **10+ endpoints** covering similarity, analogy, odd-one-out, clustering, and embeddings
- **Any language** — any ISO 639-1/2 code is accepted; models are downloaded from HuggingFace Hub on first use
- **FastText OOV support** — subword vectors for out-of-vocabulary words
- **Swagger UI** at `/apidocs`
- **Interactive examples** at `/examples`
- **Rate limiting**, **TTL caching**, **structured JSON logging**, and **Prometheus metrics**
- Docker & docker-compose ready

---

## Endpoints

### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| GET | `/languages` | List currently-loaded language models |

### Similarity

| Method | Path | Description |
|--------|------|-------------|
| GET | `/similarity` | Cosine similarity between two words (0–100 scale) |
| POST | `/similarity/batch` | Similarity for multiple word pairs |
| GET | `/most-similar` | Top-N most similar words |
| POST | `/most-similar/batch` | Most-similar for multiple words |
| GET | `/sentence-similarity` | Semantic similarity between two phrases |

### Analogy

| Method | Path | Description |
|--------|------|-------------|
| GET | `/analogy` | Solve analogies: `word1 − negative + word2 = ?` |
| GET | `/odd-one-out` | Find the word that doesn't belong in a list |

### Embeddings

| Method | Path | Description |
|--------|------|-------------|
| GET | `/word-vector` | Raw embedding vector for a word |
| GET | `/cluster` | K-means clustering of words by their embeddings |

---

## Quick Start

### Local (development)

```bash
pip install -r requirements.txt
python run.py          # starts on http://localhost:5000
```

### Docker

```bash
docker-compose up --build
```

### Gunicorn (production)

```bash
gunicorn semanticapi:app --workers 4 --preload --bind 0.0.0.0:5000
```

---

## Example Requests

```bash
# Word similarity
curl "http://localhost:5000/similarity?word1=king&word2=queen&lang=en"
# {"similarity": 71.24}

# Most similar words
curl "http://localhost:5000/most-similar?word=ocean&lang=en&topn=5"

# Analogy: king - man + woman = ?
curl "http://localhost:5000/analogy?word1=king&word2=woman&negative=man&lang=en"

# Odd one out
curl "http://localhost:5000/odd-one-out?words=breakfast,lunch,dinner,cereal&lang=en"

# Sentence similarity
curl "http://localhost:5000/sentence-similarity?text1=the+cat+sat&text2=a+dog+lay&lang=en"

# K-means clustering
curl "http://localhost:5000/cluster?words=cat,dog,car,truck&k=2&lang=en"

# Raw word vector
curl "http://localhost:5000/word-vector?word=python&lang=en"

# Batch similarity (POST)
curl -X POST http://localhost:5000/similarity/batch \
  -H "Content-Type: application/json" \
  -d '{"pairs":[{"word1":"king","word2":"queen"},{"word1":"man","word2":"woman"}],"lang":"en"}'
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDINGS_DIR` | `embeddings` | Directory for local model files |
| `PRELOAD_LANGS` | _(empty)_ | Comma-separated language codes to load at startup (e.g. `en,es`) |
| `MAX_TOPN` | `200` | Maximum allowed `topn` value |
| `MAX_WORD_LEN` | `200` | Maximum word length in characters |
| `MAX_WORDS_LIST_LEN` | `2000` | Maximum length of the `words` query parameter |
| `RATE_LIMIT_DEFAULT` | `60 per minute` | Flask-Limiter rate limit string |
| `CACHE_TTL` | `3600` | TTL (seconds) for the in-process response cache |
| `CACHE_MAXSIZE` | `1024` | Maximum entries in each cache |
| `HF_TOKEN` | _(none)_ | HuggingFace Hub token (for private/gated repos) |
| `HF_REPO_{LANG}` | `facebook/fasttext-{lang}-vectors` | Override HF repo for a language (e.g. `HF_REPO_EN`) |
| `HF_FILE_{LANG}` | `model.bin` | Override filename in repo (e.g. `HF_FILE_EN`) |
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `0` | Set to `1` to enable debug mode |

---

## Model Downloads

Models are automatically downloaded from HuggingFace Hub on first use. You can also pre-download them:

```bash
# Download English and Spanish models
python download_models.py --lang en es

# With a HuggingFace token
HF_TOKEN=hf_... python download_models.py --lang en

# Override the repository
HF_REPO_EN=my-org/my-en-model python download_models.py --lang en
```

---

## API Documentation

- **Swagger UI**: `http://localhost:5000/apidocs`
- **Interactive examples**: `http://localhost:5000/examples`

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Deployment (Render)

Set `USE_HF_DOWNLOAD=1` to download models from HuggingFace Hub during build, or leave unset to use the legacy wget approach. See `.render/build.sh` for details.


SemanticAPI is a lightweight Flask-based API for computing semantic
similarity between two words using cosine similarity over pre-trained
word embeddings.

------------------------------------------------------------------------

## ⚠️ Status

**Prototype **

-   API may change
-   Minimal validation
-   Not production hardened

------------------------------------------------------------------------

## Endpoints

### `GET /similarity`

Compute cosine similarity between two words.

#### Query Parameters

  -----------------------------------------------------------------------
  Parameter          Type          Required         Description
  ------------------ ------------- ---------------- ---------------------
  `word1`            string        Yes              First word

  `word2`            string        Yes              Second word

  `lang`             string        No               Language code (`en`,
                                                    `he`, `es`). Default:
                                                    `en`
  -----------------------------------------------------------------------

#### Success Response

``` json
{
  "similarity": 87.42
}
```

Similarity is returned as a percentage (0--100).

------------------------------------------------------------------------

### `GET /most-similar`

Return the N words most similar to a given word.

#### Query Parameters

  -----------------------------------------------------------------------
  Parameter          Type          Required         Description
  ------------------ ------------- ---------------- ---------------------
  `word`             string        Yes              Query word

  `topn`             integer       No               Number of results
                                                    (default: 10)

  `lang`             string        No               Language code. Default:
                                                    `en`
  -----------------------------------------------------------------------

#### Success Response

``` json
{
  "word": "king",
  "similar": [
    {"word": "queen", "similarity": 75.12},
    {"word": "monarch", "similarity": 68.34}
  ]
}
```

------------------------------------------------------------------------

### `GET /analogy`

Solve word-vector analogies: `word1 − negative + word2 = ?`

Example: *king − man + woman = queen* →
`?word1=king&word2=woman&negative=man`

#### Query Parameters

  -----------------------------------------------------------------------
  Parameter          Type          Required         Description
  ------------------ ------------- ---------------- ---------------------
  `word1`            string        Yes              First positive word

  `word2`            string        Yes              Second positive word

  `negative`         string        Yes              Word to subtract

  `topn`             integer       No               Number of results
                                                    (default: 5)

  `lang`             string        No               Language code. Default:
                                                    `en`
  -----------------------------------------------------------------------

#### Success Response

``` json
{
  "analogy": [
    {"word": "queen", "similarity": 72.31},
    {"word": "princess", "similarity": 65.10}
  ]
}
```

------------------------------------------------------------------------

### `GET /odd-one-out`

Identify the word that does not belong among a list of words.

#### Query Parameters

  -----------------------------------------------------------------------
  Parameter          Type          Required         Description
  ------------------ ------------- ---------------- ---------------------
  `words`            string        Yes              Comma-separated list
                                                    of words (min 3)

  `lang`             string        No               Language code. Default:
                                                    `en`
  -----------------------------------------------------------------------

#### Success Response

``` json
{
  "odd_one_out": "cereal",
  "words": ["breakfast", "cereal", "lunch", "dinner"]
}
```

------------------------------------------------------------------------

### Common Error Responses

Unsupported language:

``` json
{
  "error": "Language 'xx' not supported"
}
```

Out-of-vocabulary words:

``` json
{
  "error": "One or both words not in vocabulary"
}
```

------------------------------------------------------------------------

## Models

Embedding files are stored in the `embeddings/` directory.
Language-to-model mapping is defined in the `LANG_MODEL_PATHS` configuration
in the application code.

The application supports two ways to obtain the model files:

### Option 1 – Download from Hugging Face Hub (recommended)

Install the extra dependency and run the bundled helper script:

``` bash
pip install huggingface_hub
python download_models.py              # downloads all languages
python download_models.py --lang en    # English only
python download_models.py --lang en es # English + Spanish
```

Each language's source repository can be overridden via environment
variables:

| Variable          | Default                         | Description              |
|-------------------|---------------------------------|--------------------------|
| `HF_REPO_EN`      | `facebook/fasttext-en-vectors`  | HF repo for English      |
| `HF_FILE_EN`      | `model.bin`                     | Filename inside the repo |
| `HF_REPO_HE`      | `facebook/fasttext-he-vectors`  | HF repo for Hebrew       |
| `HF_FILE_HE`      | `model.bin`                     | Filename inside the repo |
| `HF_REPO_ES`      | `facebook/fasttext-es-vectors`  | HF repo for Spanish      |
| `HF_FILE_ES`      | `model.bin`                     | Filename inside the repo |
| `HF_TOKEN`        | *(unset)*                       | Token for private repos  |
| `EMBEDDINGS_DIR`  | `embeddings`                    | Local storage directory  |

Point to any custom repository or file:

``` bash
python download_models.py --lang en \
    --repo-id my-org/my-en-model \
    --filename vectors.vec
```

The application will also **auto-download** a missing model on first
request when `huggingface_hub` is installed and the local file does not
exist.

### Option 2 – Download FastText `.vec` files with wget (legacy)

``` bash
mkdir -p embeddings
wget -O embeddings/cc.en.300.vec.gz \
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip embeddings/cc.en.300.vec.gz
```

Repeat for other languages (`cc.he.300.vec.gz`, `cc.es.300.vec.gz`).

------------------------------------------------------------------------

## Running Locally

Install dependencies:

``` bash
pip install -r requirements.txt
```

Download models (choose one option from the **Models** section above),
then start the server:

``` bash
python app.py
```

Default port: `5000`

Override port:

``` bash
PORT=8080 python app.py
```

------------------------------------------------------------------------

## License

MIT
