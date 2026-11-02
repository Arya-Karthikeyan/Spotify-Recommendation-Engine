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
    assert len(res["songs"]) == 10
    assert seed not in [s["id"] for s in res["songs"]]


def test_nirvana_gets_rock():
    seed = client.get("/api/search", params={"q": "smells like teen spirit"}).json()[0]["id"]
    res = client.post("/api/recommend", json={"seeds": [seed], "n": 10}).json()["songs"]
    assert sum("rock" in s["genres"] for s in res) >= 7


def test_bad_requests():
    assert client.post("/api/recommend", json={"seeds": ["fake"]}).status_code == 400
    assert client.post("/api/recommend", json={"seeds": []}).status_code == 400
    assert client.get("/api/search", params={"q": "a"}).json() == []


def test_taste_profile():
    seed = client.get("/api/search", params={"q": "basket case - green day"}).json()[0]["id"]
    taste = client.post("/api/recommend", json={"seeds": [seed], "n": 10}).json()["taste"]
    assert taste["decade"] == "1990s"
    assert "rock" in taste["genres"]
    for key in ["energy", "danceability", "valence", "acousticness", "tempo", "popularity"]:
        assert 0 <= taste["seeds"][key] <= 1
        assert 0 <= taste["recs"][key] <= 1


def test_mood_changes_results():
    seed = client.get("/api/search", params={"q": "smells like teen spirit"}).json()[0]["id"]
    normal = client.post("/api/recommend", json={"seeds": [seed], "n": 20}).json()["taste"]["recs"]
    calm = client.post("/api/recommend", json={"seeds": [seed], "n": 20,
                                               "mood": {"energy": -1, "acousticness": 1}}).json()["taste"]["recs"]
    assert calm["energy"] < normal["energy"] - 0.1
    assert calm["acousticness"] > normal["acousticness"] + 0.1
