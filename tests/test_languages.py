def test_languages(client):
    r = client.get("/languages")
    assert r.status_code == 200
    data = r.get_json()
    assert "loaded" in data
    assert "note" in data
