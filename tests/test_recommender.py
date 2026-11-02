import numpy as np
import pandas as pd
import pytest

from recommender.data import build_songs, clean_text, get_playlists
from recommender.engine import Recommender
from recommender.spotify_api import match_to_dataset

ROCK = dict(energy=0.95, loudness=-3.0, acousticness=0.01, danceability=0.4, valence=0.4)
CHILL = dict(energy=0.2, loudness=-14.0, acousticness=0.9, danceability=0.5, valence=0.3)


def make_row(track_id, name, artist, genre, subgenre, playlist="p1", popularity=50, **features):
    row = dict(track_id=track_id, track_name=name, track_artist=artist, track_popularity=popularity,
               track_album_release_date="2015-01-01", playlist_id=playlist,
               playlist_genre=genre, playlist_subgenre=subgenre,
               danceability=0.5, energy=0.5, loudness=-6.0, speechiness=0.05, acousticness=0.1,
               instrumentalness=0.0, liveness=0.1, valence=0.5, tempo=120.0)
    row.update(features)
    return row


@pytest.fixture
def raw():
    return pd.DataFrame([
        make_row("r1", "Loud One", "Band A", "rock", "hard rock", **ROCK),
        make_row("r2", "Loud Two", "Band B", "rock", "hard rock", playlist="p3", **ROCK),
        make_row("r3", "Loud Three", "Band A", "rock", "hard rock", playlist="p3", **ROCK),
        make_row("c1", "Soft One", "Singer C", "pop", "indie poptimism", **CHILL),
        make_row("c2", "Soft Two", "Singer D", "pop", "indie poptimism", playlist="p4", **CHILL),
        # duplicate of r2 with a different id
        make_row("r2b", "Loud Two (Remastered)", "Band B", "rock", "hard rock", playlist="p3", popularity=10, **ROCK),
        # r1 is in a second playlist too
        make_row("r1", "Loud One", "Band A", "edm", "big room", playlist="p2", **ROCK),
    ])


@pytest.fixture
def rec(raw):
    songs = build_songs(raw)
    return Recommender(songs, get_playlists(raw, songs))


def test_clean_text():
    assert clean_text("Beyoncé (feat. JAY-Z)") == "beyonce"
    assert clean_text("Song - 2011 Remastered Version") == "song"


def test_build_songs(raw):
    songs = build_songs(raw)
    assert songs.loc["r1", "genres"] == ["edm", "rock"]
    assert "r2b" not in songs.index  # duplicate removed
    assert "r2" in songs.index


def test_vectors_are_normalised(rec):
    assert np.allclose(np.linalg.norm(rec.vectors, axis=1), 1, atol=1e-6)


def test_similar_songs_come_first(rec):
    results = rec.recommend(["r1"], n=4, collab=0, popularity=0, max_per_artist=5)
    ids = [r.track_id for r in results]
    assert "r1" not in ids
    assert set(ids[:2]) == {"r2", "r3"}


def test_max_per_artist_and_new_artists(rec):
    results = rec.recommend(["r2"], n=4, max_per_artist=1)
    artists = [r.artist for r in results]
    assert len(artists) == len(set(artists))
    results = rec.recommend(["r1"], n=4, new_artists_only=True)
    assert "Band A" not in [r.artist for r in results]


def test_genre_filter(rec):
    results = rec.recommend(["r1"], n=4, genres=["pop"])
    assert {r.track_id for r in results} == {"c1", "c2"}


def test_playlist_mates_get_boosted(rec):
    # c1 shares playlist p1 with r1 but sounds totally different
    sound_only = [r.track_id for r in rec.recommend(["r1"], n=4, collab=0, popularity=0, new_artists_only=True)]
    with_collab = rec.recommend(["r1"], n=4, collab=0.9, popularity=0, new_artists_only=True)
    assert sound_only[0] == "r2"
    assert with_collab[0].track_id == "c1"
    assert "playlisted with Loud One" in with_collab[0].reasons


def test_dislikes(rec):
    before = rec.recommend(["r1", "c1"], n=4, collab=0)
    after = rec.recommend(["r1", "c1"], n=4, collab=0, dislikes=["c2"])
    assert "c2" not in [r.track_id for r in after]
    assert before != after


def test_find(rec):
    assert rec.find("loud two - band b") == "r2"
    assert rec.find("Soft One by Singer C") == "c1"
    assert rec.find("c2") == "c2"
    with pytest.raises(KeyError):
        rec.find("some song that isnt there - nobody")


def test_match_spotify_tracks(rec):
    tracks = [
        {"id": "r3", "name": "x", "artist": "y"},
        {"id": "abc", "name": "Soft Two", "artist": "Singer D"},
        {"id": "xyz", "name": "Not Here", "artist": "Nobody"},
    ]
    found, missing = match_to_dataset(rec, tracks)
    assert found == ["r3", "c2"]
    assert missing[0]["name"] == "Not Here"


def test_mood_bonus(rec):
    # seeds are one rock song and one chill song, so the mood sliders should decide
    # which kind comes first
    kw = dict(n=1, collab=0, popularity=0)
    calm = rec.recommend(["r1", "c1"], mood={"energy": -1, "acousticness": 1}, **kw)
    loud = rec.recommend(["r1", "c1"], mood={"energy": 1, "acousticness": -1}, **kw)
    assert calm[0].track_id == "c2"
    assert loud[0].track_id in {"r2", "r3"}
