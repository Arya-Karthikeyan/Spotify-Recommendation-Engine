import pytest
from fastapi.testclient import TestClient

from recommender.app import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def need_data():
    from recommender.data import DATA_PATH
    if not DATA_PATH.exists():
        pytest.skip("dataset not downloaded")


def test_homepage():
    assert "Crate Digger" in client.get("/").text


def test_search_then_recommend():
    songs = client.get("/api/search", params={"q": "basket case - green day"}).json()
    assert songs[0]["artist"] == "Green Day"
    seed = songs[0]["id"]
    res = client.post("/api/recommend", json={"seeds": [seed], "n": 10}).json()
    assert len(res) == 10
    assert seed not in [s["id"] for s in res]


def test_nirvana_gets_rock():
    seed = client.get("/api/search", params={"q": "smells like teen spirit"}).json()[0]["id"]
    res = client.post("/api/recommend", json={"seeds": [seed], "n": 10}).json()
    assert sum("rock" in s["genres"] for s in res) >= 7


def test_bad_requests():
    assert client.post("/api/recommend", json={"seeds": ["fake"]}).status_code == 400
    assert client.post("/api/recommend", json={"seeds": []}).status_code == 400
    assert client.get("/api/search", params={"q": "a"}).json() == []
