# Evaluation: can the recommender "finish" a playlist?
#
# For 150 random playlists: hide half the songs, give the other half as seeds,
# and check how many hidden songs show up in the top 50 recommendations (recall@50).
#
# The test playlists are left out completely when building the model, including
# the genre labels (in this dataset genres come from the playlists, so otherwise
# the model would be cheating).

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.data import build_songs, get_playlists, load_raw
from recommender.engine import Recommender

K = 50
N_TEST = 150
rng = np.random.default_rng(0)

raw = load_raw()
songs = build_songs(raw)
playlists = get_playlists(raw, songs)

tracks_in = playlists.groupby("playlist_id")["track_id"].apply(list)
big_enough = tracks_in[tracks_in.apply(len) >= 10].index
test_ids = set(rng.choice(big_enough, N_TEST, replace=False))
train = playlists[~playlists["playlist_id"].isin(test_ids)]

# rebuild genre labels using training playlists only
info = raw.drop_duplicates("playlist_id").set_index("playlist_id")
train_info = train.join(info[["playlist_genre", "playlist_subgenre"]], on="playlist_id")
genres = train_info.groupby("track_id")["playlist_genre"].apply(lambda x: sorted(set(x)))
subgenres = train_info.groupby("track_id")["playlist_subgenre"].apply(lambda x: sorted(set(x)))
songs = songs.copy()
songs["genres"] = [genres.get(t, []) for t in songs.index]
songs["subgenres"] = [subgenres.get(t, []) for t in songs.index]

print("building models...")
hybrid = Recommender(songs, train)
content = Recommender(songs)
audio_only = Recommender(songs, genre_weight=0, subgenre_weight=0)

models = {
    "hybrid (collab 0.2)": (hybrid, {"collab": 0.2}),
    "hybrid (collab 0.4) *": (hybrid, {"collab": 0.4}),
    "hybrid (collab 0.6)": (hybrid, {"collab": 0.6}),
    "hybrid (collab 0.8)": (hybrid, {"collab": 0.8}),
    "content only": (content, {}),
    "audio + year only": (audio_only, {}),
}
scores = {name: [] for name in list(models) + ["most popular", "random"]}
by_popularity = songs["track_popularity"].sort_values(ascending=False).index

for pid in sorted(test_ids):
    tracks = np.array(tracks_in[pid])
    rng.shuffle(tracks)
    seeds = list(tracks[: len(tracks) // 2])
    hidden = set(tracks[len(tracks) // 2:])

    for name, (model, options) in models.items():
        recs = model.recommend(seeds, n=K, **options)
        scores[name].append(len(hidden & {r.track_id for r in recs}) / len(hidden))

    popular = [t for t in by_popularity if t not in seeds][:K]
    scores["most popular"].append(len(hidden & set(popular)) / len(hidden))
    others = [t for t in songs.index if t not in seeds]
    scores["random"].append(len(hidden & set(rng.choice(others, K, replace=False))) / len(hidden))

print(f"\nrecall@{K} on {N_TEST} held out playlists\n")
baseline = np.mean(scores["random"])
for name, s in scores.items():
    print(f"{name:<24} {np.mean(s):.3f}  ({np.mean(s) / baseline:.0f}x random)")
