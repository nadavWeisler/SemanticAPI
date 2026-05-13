def test_similarity_happy_path(client):
    r = client.get("/similarity?word1=king&word2=queen&lang=en")
    assert r.status_code == 200
    data = r.get_json()
    assert "similarity" in data
    assert 0 <= data["similarity"] <= 100


def test_similarity_missing_words(client):
    r = client.get("/similarity?word1=king&lang=en")
    assert r.status_code == 400


def test_similarity_oov_word(client):
    r = client.get("/similarity?word1=king&word2=zzzzunknown&lang=en")
    assert r.status_code == 400
    assert "not in vocabulary" in r.get_json()["error"]


def test_similarity_invalid_lang(client):
    r = client.get("/similarity?word1=king&word2=queen&lang=xx99")
    assert r.status_code == 400


def test_similarity_batch(client):
    payload = {"pairs": [{"word1": "king", "word2": "queen"}, {"word1": "man", "word2": "woman"}], "lang": "en"}
    r = client.post("/similarity/batch", json=payload)
    assert r.status_code == 200
    results = r.get_json()["results"]
    assert len(results) == 2
    for res in results:
        assert "similarity" in res


def test_similarity_batch_missing_field(client):
    r = client.post("/similarity/batch", json={"lang": "en"})
    assert r.status_code == 400
