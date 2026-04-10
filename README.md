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
