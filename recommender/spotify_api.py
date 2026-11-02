# Uses the Spotify API (through spotipy) to get your top/recent/saved songs and to
# save the recommendations as a playlist.
#
# Spotify removed the /recommendations and /audio-features endpoints for new apps
# in Nov 2024 so the actual recommending is all done locally.
#
# You need to make an app at https://developer.spotify.com/dashboard and set:
#   SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI (redirect uri is only for the cli)

import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth

SCOPES = "user-top-read user-read-recently-played user-library-read playlist-modify-private"


def connect():
    """For the command line. Logs in through the terminal and caches the token in .cache"""
    return spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES, open_browser=False))


def make_auth(redirect_uri, cache_handler):
    """For the web app. Each browser session gets its own token cache."""
    return SpotifyOAuth(scope=SCOPES, redirect_uri=redirect_uri, cache_handler=cache_handler,
                        open_browser=False, show_dialog=True)


def simplify(track):
    if not track or not track.get("id"):
        return None
    return {"id": track["id"], "name": track["name"], "artist": track["artists"][0]["name"]}


def get_tracks(sp, source="top", time_range="medium_term", playlist=None):
    if source == "top":
        items = sp.current_user_top_tracks(limit=50, time_range=time_range)["items"]
    elif source == "recent":
        items = [i["track"] for i in sp.current_user_recently_played(limit=50)["items"]]
    elif source == "saved":
        items = [i["track"] for i in sp.current_user_saved_tracks(limit=50)["items"]]
    elif source == "playlist":
        items = [i.get("track") for i in sp.playlist_items(playlist, limit=100)["items"]]
    else:
        raise ValueError("unknown source " + source)
    return [t for t in map(simplify, items) if t]


def match_to_dataset(rec, tracks):
    """Find each Spotify track in our dataset, by id first and then by name + artist."""
    found, missing = [], []
    for t in tracks:
        tid = t["id"] if t["id"] in rec.row_of else rec.find_exact(t["name"], t["artist"])
        if tid is None:
            missing.append(t)
        elif tid not in found:
            found.append(tid)
    return found, missing


def is_configured():
    return bool(os.environ.get("SPOTIPY_CLIENT_ID") and os.environ.get("SPOTIPY_CLIENT_SECRET"))


def save_playlist(sp, name, track_ids):
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, name, public=False,
                                       description="made with spotify-recommendation-engine")
    for i in range(0, len(track_ids), 100):  # api only takes 100 at a time
        sp.playlist_add_items(playlist["id"], track_ids[i:i + 100])
    return playlist["external_urls"]["spotify"]
