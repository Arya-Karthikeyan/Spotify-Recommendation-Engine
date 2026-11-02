import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "spotify_songs.csv"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def clean_text(text):
    """Lowercase and strip accents/punctuation so titles can be matched.
    Also removes stuff like '(feat. X)' and '- Remastered 2011'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"\s*[\(\[].*?[\)\]]", "", text)
    text = re.sub(r"\s+-\s+.*(remaster|version|remix|edit|live|mono|stereo).*$", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def load_raw(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found, run scripts/download_data.py first")
    df = pd.read_csv(path)
    return df.dropna(subset=["track_name", "track_artist"])


def build_songs(raw):
    """The csv has one row per (song, playlist) so the same song can show up
    lots of times. This makes it one row per song and keeps all its genres."""
    genres = raw.groupby("track_id")["playlist_genre"].agg(lambda x: sorted(set(x)))
    subgenres = raw.groupby("track_id")["playlist_subgenre"].agg(lambda x: sorted(set(x)))

    songs = raw.drop_duplicates("track_id").set_index("track_id")
    songs["genres"] = genres
    songs["subgenres"] = subgenres

    year = pd.to_numeric(songs["track_album_release_date"].str[:4], errors="coerce")
    songs["year"] = year.fillna(year.median()).astype(int)

    songs["name_key"] = songs["track_name"].apply(clean_text)
    songs["artist_key"] = songs["track_artist"].apply(clean_text)

    # same song is often on the single + the album + a compilation with different ids,
    # just keep the most popular one
    songs = songs.sort_values("track_popularity", ascending=False)
    songs = songs.drop_duplicates(["name_key", "artist_key"]).sort_index()

    cols = ["track_name", "track_artist", "track_popularity", "year", "genres", "subgenres",
            "name_key", "artist_key"] + AUDIO_FEATURES
    return songs[cols]


def get_playlists(raw, songs):
    """Returns a dataframe of (playlist_id, track_id) using the deduplicated song ids."""
    lookup = {(n, a): tid for tid, n, a in zip(songs.index, songs["name_key"], songs["artist_key"])}
    ids = [lookup.get((clean_text(n), clean_text(a)))
           for n, a in zip(raw["track_name"], raw["track_artist"])]
    df = pd.DataFrame({"playlist_id": raw["playlist_id"].values, "track_id": ids})
    return df.dropna().drop_duplicates().reset_index(drop=True)
