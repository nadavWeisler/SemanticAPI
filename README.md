# SemanticAPI

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
