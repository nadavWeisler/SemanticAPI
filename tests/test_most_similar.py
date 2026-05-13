def test_most_similar_happy_path(client):
    r = client.get("/most-similar?word=king&lang=en&topn=3")
    assert r.status_code == 200
    data = r.get_json()
    assert data["word"] == "king"
    assert len(data["similar"]) <= 3


def test_most_similar_oov(client):
    r = client.get("/most-similar?word=zzzzunknown&lang=en")
    assert r.status_code == 400


def test_most_similar_bad_topn(client):
    r = client.get("/most-similar?word=king&lang=en&topn=-1")
    assert r.status_code == 400


def test_most_similar_batch(client):
    payload = {"words": ["king", "queen"], "lang": "en", "topn": 3}
    r = client.post("/most-similar/batch", json=payload)
    assert r.status_code == 200
    results = r.get_json()["results"]
    assert len(results) == 2
