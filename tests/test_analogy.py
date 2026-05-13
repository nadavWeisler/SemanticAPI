def test_analogy_happy_path(client):
    r = client.get("/analogy?word1=king&word2=woman&negative=man&lang=en")
    assert r.status_code == 200
    data = r.get_json()
    assert "analogy" in data
    assert isinstance(data["analogy"], list)


def test_analogy_missing_params(client):
    r = client.get("/analogy?word1=king&lang=en")
    assert r.status_code == 400


def test_analogy_oov(client):
    r = client.get("/analogy?word1=king&word2=zzz&negative=man&lang=en")
    assert r.status_code == 400
