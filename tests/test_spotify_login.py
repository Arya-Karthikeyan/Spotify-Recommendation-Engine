# tests for the spotify parts of the web app, using a fake spotify client
# so no real account is needed
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from recommender import app as app_module

client = TestClient(app_module.app)


def track(id, name, artist):
    return {"id": id, "name": name, "artists": [{"name": artist}]}


class FakeSpotify:
    def __init__(self):
        self.saved = None

    def current_user_top_tracks(self, limit, time_range):
        return {"items": [
            track("not-a-real-id", "Basket Case", "Green Day"),  # should match by name
            track("abc", "Some 2024 Song", "Someone"),  # not in the dataset
        ]}

    def current_user(self):
        return {"id": "arya", "display_name": "Arya"}

    def user_playlist_create(self, user, name, public, description):
        self.saved = {"name": name, "tracks": []}
        return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}

    def playlist_add_items(self, playlist_id, ids):
        self.saved["tracks"] += ids


@pytest.fixture
def fake(monkeypatch):
    sp = FakeSpotify()
    monkeypatch.setattr(app_module, "get_spotify", lambda request: sp)
    return sp


def test_not_connected():
    assert client.get("/api/me").json()["connected"] is False
    assert client.get("/api/spotify/songs").status_code == 401
    assert client.post("/api/spotify/save", json={"name": "x", "track_ids": ["a"]}).status_code == 401


def test_login_redirects_to_spotify(monkeypatch):
    monkeypatch.setenv("SPOTIPY_CLIENT_ID", "test-id")
    monkeypatch.setenv("SPOTIPY_CLIENT_SECRET", "test-secret")
    res = client.get("/login", follow_redirects=False)
    url = urlparse(res.headers["location"])
    assert url.netloc == "accounts.spotify.com"
    params = parse_qs(url.query)
    assert params["client_id"] == ["test-id"]
    assert "user-top-read" in params["scope"][0]
    assert "session" in res.cookies

    # coming back with the wrong state should fail
    res = client.get("/callback", params={"code": "x", "state": "wrong"}, follow_redirects=False)
    assert res.headers["location"] == "/?spotify=failed"


def test_login_without_keys(monkeypatch):
    monkeypatch.delenv("SPOTIPY_CLIENT_ID", raising=False)
    assert client.get("/login", follow_redirects=False).status_code == 500
    assert client.get("/api/me").json()["configured"] is False


def test_my_songs(fake):
    data = client.get("/api/spotify/songs", params={"source": "top"}).json()
    assert data["total"] == 2
    assert [s["name"] for s in data["songs"]] == ["Basket Case"]


def test_save_playlist(fake):
    res = client.post("/api/spotify/save", json={"name": "My mix", "track_ids": ["a", "b"]})
    assert res.json()["url"].endswith("pl1")
    assert fake.saved == {"name": "My mix", "tracks": ["a", "b"]}
