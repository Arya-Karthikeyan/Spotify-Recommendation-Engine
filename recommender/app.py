# Web app. Run with:  uvicorn recommender.app:app
import os
import secrets
from pathlib import Path

import spotipy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from spotipy.cache_handler import MemoryCacheHandler

from . import spotify_api
from .engine import Recommender

app = FastAPI()
rec = None

# this has to match a redirect uri added in the spotify developer dashboard
REDIRECT_URI = os.environ.get("WEB_REDIRECT_URI", "http://127.0.0.1:8000/callback")

# logged in users, keyed by a random id stored in a cookie. it's just kept in memory
# so everyone has to log in again if the server restarts, which is fine for this
sessions = {}


def get_model():
    # load lazily so importing the app is quick
    global rec
    if rec is None:
        rec = Recommender.from_csv()
    return rec


def song_json(tid):
    row = get_model().songs.loc[tid]
    return {
        "id": tid,
        "name": row["track_name"],
        "artist": row["track_artist"],
        "year": int(row["year"]),
        "tempo": round(float(row["tempo"])),
        "genres": list(row["genres"]),
        "energy": float(row["energy"]),
        "danceability": float(row["danceability"]),
        "valence": float(row["valence"]),
        "acousticness": float(row["acousticness"]),
    }


class RecRequest(BaseModel):
    seeds: list[str]
    dislikes: list[str] = []
    n: int = 20
    diversity: float = 0.3
    popularity: float = 0.1
    collab: float = 0.4
    new_artists: bool = False
    genres: list[str] = []


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/search")
def search(q: str):
    if len(q.strip()) < 2:
        return []
    return [song_json(tid) for tid in get_model().search(q, limit=8).index]


@app.post("/api/recommend")
def recommend(req: RecRequest):
    model = get_model()
    if not req.seeds:
        raise HTTPException(400, "need at least one seed")
    for tid in req.seeds + req.dislikes:
        if tid not in model.row_of:
            raise HTTPException(400, f"unknown track id {tid}")
    results = model.recommend(
        req.seeds,
        n=min(req.n, 100),
        dislikes=req.dislikes,
        diversity=req.diversity,
        popularity=req.popularity,
        collab=req.collab,
        new_artists_only=req.new_artists,
        genres=req.genres or None,
    )
    out = []
    for r in results:
        song = song_json(r.track_id)
        song["match"] = round(r.sound_match, 2)
        song["reasons"] = r.reasons
        out.append(song)
    return out


# ---- spotify login ----

def get_spotify(request):
    """Returns a logged in spotipy client for this browser, or None."""
    session = sessions.get(request.cookies.get("session"))
    if session is None or session["cache"].get_cached_token() is None:
        return None
    auth = spotify_api.make_auth(REDIRECT_URI, session["cache"])
    return spotipy.Spotify(auth_manager=auth)  # spotipy refreshes the token when it expires


def require_spotify(request):
    sp = get_spotify(request)
    if sp is None:
        raise HTTPException(401, "not connected to spotify")
    return sp


@app.get("/login")
def login():
    if not spotify_api.is_configured():
        raise HTTPException(500, "set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET first")
    session_id = secrets.token_urlsafe(16)
    state = secrets.token_urlsafe(16)  # checked in /callback to stop CSRF
    sessions[session_id] = {"cache": MemoryCacheHandler(), "state": state, "name": None}
    url = spotify_api.make_auth(REDIRECT_URI, sessions[session_id]["cache"]).get_authorize_url(state=state)
    response = RedirectResponse(url)
    response.set_cookie("session", session_id, httponly=True, samesite="lax")
    return response


@app.get("/callback")
def callback(request: Request, code: str = None, state: str = None, error: str = None):
    session = sessions.get(request.cookies.get("session"))
    if error or not code or session is None or session["state"] != state:
        return RedirectResponse("/?spotify=failed")
    auth = spotify_api.make_auth(REDIRECT_URI, session["cache"])
    auth.get_access_token(code, as_dict=False, check_cache=False)
    sp = spotipy.Spotify(auth_manager=auth)
    me = sp.current_user()
    session["name"] = me.get("display_name") or me["id"]
    return RedirectResponse("/?spotify=connected")


@app.get("/logout")
def logout(request: Request):
    sessions.pop(request.cookies.get("session"), None)
    response = RedirectResponse("/")
    response.delete_cookie("session")
    return response


@app.get("/api/me")
def me(request: Request):
    session = sessions.get(request.cookies.get("session"))
    connected = get_spotify(request) is not None
    name = session["name"] if connected and session else None
    return {"configured": spotify_api.is_configured(), "connected": connected, "name": name}


@app.get("/api/spotify/songs")
def my_songs(request: Request, source: str = "top", time_range: str = "short_term"):
    sp = require_spotify(request)
    if source not in ["top", "recent", "saved"]:
        raise HTTPException(400, "source should be top, recent or saved")
    tracks = spotify_api.get_tracks(sp, source, time_range)
    found, missing = spotify_api.match_to_dataset(get_model(), tracks)
    # 25 seeds is plenty, after that the taste profile just turns into an average of everything
    return {"total": len(tracks), "songs": [song_json(tid) for tid in found[:25]]}


class SaveRequest(BaseModel):
    name: str
    track_ids: list[str]


@app.post("/api/spotify/save")
def save(request: Request, req: SaveRequest):
    sp = require_spotify(request)
    if not req.track_ids:
        raise HTTPException(400, "no songs to save")
    url = spotify_api.save_playlist(sp, req.name.strip() or "Crate Digger mix", req.track_ids)
    return {"url": url}
