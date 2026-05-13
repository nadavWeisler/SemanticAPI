def test_cluster_happy_path(client):
    r = client.get("/cluster?words=king,queen,man,woman,cat,dog&k=2&lang=en")
    assert r.status_code == 200
    data = r.get_json()
    assert data["k"] == 2
    assert len(data["clusters"]) == 2
    total = sum(len(c) for c in data["clusters"])
    assert total == 6


def test_cluster_k_too_large(client):
    r = client.get("/cluster?words=king,queen&k=5&lang=en")
    assert r.status_code == 400


def test_cluster_bad_k(client):
    r = client.get("/cluster?words=king,queen,man&k=abc&lang=en")
    assert r.status_code == 400
