# Web app. Run with:  uvicorn recommender.app:app
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .engine import Recommender

app = FastAPI()
rec = None


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
