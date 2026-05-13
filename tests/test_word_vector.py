def test_word_vector_happy_path(client):
    r = client.get("/word-vector?word=king&lang=en")
    assert r.status_code == 200
    data = r.get_json()
    assert "vector" in data
    assert data["dimensions"] == 10
    assert len(data["vector"]) == 10


def test_word_vector_missing_word(client):
    r = client.get("/word-vector?lang=en")
    assert r.status_code == 400


def test_word_vector_oov(client):
    r = client.get("/word-vector?word=zzzzunknown&lang=en")
    assert r.status_code == 400
