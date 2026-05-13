def test_odd_one_out_happy_path(client):
    r = client.get("/odd-one-out?words=breakfast,lunch,dinner,cereal&lang=en")
    assert r.status_code == 200
    data = r.get_json()
    assert "odd_one_out" in data
    assert "ranking" in data
    assert len(data["ranking"]) == 4


def test_odd_one_out_too_few(client):
    r = client.get("/odd-one-out?words=king,queen&lang=en")
    assert r.status_code == 400


def test_odd_one_out_oov(client):
    r = client.get("/odd-one-out?words=king,queen,zzz&lang=en")
    assert r.status_code == 400
