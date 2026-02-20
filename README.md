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

## Endpoint

### `GET /similarity`

Compute cosine similarity between two words.

### Query Parameters

  -----------------------------------------------------------------------
  Parameter          Type          Required         Description
  ------------------ ------------- ---------------- ---------------------
  `word1`            string        Yes              First word

  `word2`            string        Yes              Second word

  `lang`             string        No               Language code (`en`,
                                                    `he`, `es`). Default:
                                                    `en`
  -----------------------------------------------------------------------

------------------------------------------------------------------------

### Success Response

``` json
{
  "similarity": 87.42
}
```

Similarity is returned as a percentage (0--100).

------------------------------------------------------------------------

### Error Responses

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

Embedding files are expected in:

    embeddings/

Language-to-model mapping is defined in the `LANG_MODELS` configuration
in the application code.

------------------------------------------------------------------------

## Running Locally

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start the server:

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
