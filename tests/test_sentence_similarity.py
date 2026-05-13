def test_sentence_similarity_happy_path(client):
    r = client.get("/sentence-similarity?text1=king man&text2=queen woman&lang=en")
    assert r.status_code == 200
    data = r.get_json()
    assert "similarity" in data
    assert -100 <= data["similarity"] <= 100


def test_sentence_similarity_missing_text(client):
    r = client.get("/sentence-similarity?text1=king&lang=en")
    assert r.status_code == 400
