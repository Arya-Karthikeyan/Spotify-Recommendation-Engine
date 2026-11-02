from collections import Counter
from dataclasses import dataclass, field
from difflib import get_close_matches

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from .collab import PlaylistSimilarity
from .data import AUDIO_FEATURES, DATA_PATH, build_songs, clean_text, get_playlists, load_raw

# how much each audio feature counts. energy/valence/danceability matter most for
# how a song "feels", liveness barely matters. got these by trying a few values
FEATURE_WEIGHTS = {
    "danceability": 1.0,
    "energy": 1.2,
    "loudness": 0.6,
    "speechiness": 0.8,
    "acousticness": 1.0,
    "instrumentalness": 0.8,
    "liveness": 0.3,
    "valence": 1.0,
    "tempo": 0.6,
}

# features worth mentioning in the "why" text. instrumentalness and liveness are ~0
# for almost every song so saying "similar instrumentalness" means nothing
EXPLAIN_FEATURES = ["danceability", "energy", "valence", "acousticness", "tempo", "speechiness", "loudness"]


@dataclass
class Recommendation:
    track_id: str
    name: str
    artist: str
    score: float
    sound_match: float
    reasons: list = field(default_factory=list)


def normalize(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


class Recommender:
    def __init__(self, songs, playlists=None, genre_weight=0.6, subgenre_weight=0.5, year_weight=0.3):
        self.songs = songs
        self.ids = list(songs.index)
        self.row_of = {tid: i for i, tid in enumerate(self.ids)}
        self.artists = songs["artist_key"].values
        self.unique_names = songs["name_key"].unique().tolist()

        # each song becomes one vector: audio features + genres + subgenres + year.
        # the vectors are normalised so a dot product = cosine similarity
        self.audio_z = StandardScaler().fit_transform(songs[AUDIO_FEATURES])
        weights = np.array([FEATURE_WEIGHTS[f] for f in AUDIO_FEATURES])
        audio = self.audio_z * weights / np.sqrt(len(AUDIO_FEATURES))

        genres = normalize(MultiLabelBinarizer().fit_transform(songs["genres"]).astype(float))
        subgenres = normalize(MultiLabelBinarizer().fit_transform(songs["subgenres"]).astype(float))
        year = StandardScaler().fit_transform(songs[["year"]])

        vectors = np.hstack([audio, genre_weight * genres, subgenre_weight * subgenres, year_weight * year])
        self.vectors = normalize(vectors)
        self.popularity = songs["track_popularity"].values / 100

        self.playlists = PlaylistSimilarity(playlists, songs) if playlists is not None else None

    @classmethod
    def from_csv(cls, path=DATA_PATH):
        raw = load_raw(path)
        songs = build_songs(raw)
        return cls(songs, get_playlists(raw, songs))

    # ---- finding songs ----

    def search(self, query, limit=10):
        """query can be 'title', 'title - artist' or 'title by artist'"""
        title, artist = split_query(query)
        songs = self.songs
        matches = songs[songs["name_key"].str.contains(title, regex=False)]
        if matches.empty:
            # probably a typo, try fuzzy matching on the title
            close = get_close_matches(title, self.unique_names, n=limit, cutoff=0.75)
            matches = songs[songs["name_key"].isin(close)]
        if artist:
            matches = matches[matches["artist_key"].str.contains(artist, regex=False)]

        # exact title matches first, then most popular
        exact = matches["name_key"] == title
        order = exact.astype(int) * 1000 + matches["track_popularity"]
        return matches.loc[order.sort_values(ascending=False).index[:limit]]

    def find(self, query):
        """Turn a search string (or a track id) into a track id."""
        if query in self.row_of:
            return query
        results = self.search(query, limit=1)
        if results.empty:
            raise KeyError(f"couldn't find {query!r} (the dataset only goes up to Jan 2020)")
        return results.index[0]

    def find_exact(self, name, artist):
        """Used for matching songs from the Spotify API."""
        n, a = clean_text(name), clean_text(artist)
        matches = self.songs[(self.songs["name_key"] == n) & (self.songs["artist_key"].str.contains(a, regex=False))]
        return matches.index[0] if len(matches) else None

    # ---- recommending ----

    def recommend(self, seeds, n=20, dislikes=None, seed_weights=None, diversity=0.3, popularity=0.1,
                  collab=0.4, max_per_artist=2, new_artists_only=False, genres=None, exclude=None):
        """
        seeds          list of track ids you like
        dislikes       track ids to move away from
        diversity      0 = just the closest songs, 1 = spread out as much as possible
        popularity     how much to boost popular songs
        collab         0 = only sound similarity, 1 = only playlist co-occurrence
        """
        if not seeds:
            raise ValueError("need at least one seed song")
        dislikes = dislikes or []
        seed_rows = [self.row_of[s] for s in seeds]
        if seed_weights is None:
            seed_weights = np.ones(len(seeds))
        seed_weights = np.asarray(seed_weights, dtype=float)

        # taste profile = weighted average of the seed vectors, minus half the dislikes
        profile = (self.vectors[seed_rows] * seed_weights[:, None]).sum(axis=0) / seed_weights.sum()
        if dislikes:
            profile -= 0.5 * self.vectors[[self.row_of[d] for d in dislikes]].mean(axis=0)
        profile = normalize(profile)

        sound = self.vectors @ profile
        match = sound
        song_cf = None
        if self.playlists is not None and collab > 0:
            song_cf, artist_cf = self.playlists.scores(seed_rows, seed_weights)
            match = (1 - collab) * sound + collab * (0.6 * song_cf + 0.4 * artist_cf)
        scores = (1 - popularity) * match + popularity * self.popularity

        # remove songs we shouldn't recommend
        allowed = np.ones(len(self.ids), dtype=bool)
        for tid in set(seeds) | set(dislikes) | set(exclude or []):
            if tid in self.row_of:
                allowed[self.row_of[tid]] = False
        seed_songs = self.songs.iloc[seed_rows]
        allowed &= ~self.songs["name_key"].isin(seed_songs["name_key"]).values  # other versions of a seed
        if new_artists_only:
            allowed &= ~self.songs["artist_key"].isin(seed_songs["artist_key"]).values
        if genres:
            allowed &= self.songs["genres"].apply(lambda g: any(x in genres for x in g)).values
        scores[~allowed] = -np.inf

        # only bother re-ranking the top 300
        candidates = [i for i in np.argsort(-scores)[:300] if allowed[i]]
        picked = self.pick_diverse(candidates, scores, n, diversity, max_per_artist)

        results = []
        for i in picked:
            row = self.songs.iloc[i]
            results.append(Recommendation(
                track_id=self.ids[i],
                name=row["track_name"],
                artist=row["track_artist"],
                score=float(scores[i]),
                sound_match=float(sound[i]),
                reasons=self.explain(i, seed_rows, song_cf),
            ))
        return results

    def pick_diverse(self, candidates, scores, n, diversity, max_per_artist):
        """Maximal marginal relevance: each step pick the song with the best score
        minus how similar it is to songs already picked. Stops the list being
        20 versions of the same song."""
        candidates = np.array(candidates)
        if len(candidates) == 0:
            return []
        similarity_to_picked = np.zeros(len(candidates))
        used = np.zeros(len(candidates), dtype=bool)
        per_artist = Counter()
        picked = []

        while len(picked) < n:
            mmr = (1 - diversity) * scores[candidates] - diversity * similarity_to_picked
            best = None
            for j in np.argsort(-mmr):
                if not used[j] and per_artist[self.artists[candidates[j]]] < max_per_artist:
                    best = j
                    break
            if best is None:
                break
            i = candidates[best]
            used[best] = True
            picked.append(i)
            per_artist[self.artists[i]] += 1
            similarity_to_picked = np.maximum(similarity_to_picked, self.vectors[candidates] @ self.vectors[i])
        return picked

    def explain(self, i, seed_rows, song_cf):
        reasons = []
        if song_cf is not None and song_cf[i] > 0.05:
            seed = self.playlists.best_seed(i, seed_rows)
            if seed is not None:
                reasons.append("playlisted with " + self.songs.iloc[seed]["track_name"])

        seed_styles = set()
        for r in seed_rows:
            seed_styles.update(self.songs.iloc[r]["subgenres"])
        shared = sorted(seed_styles & set(self.songs.iloc[i]["subgenres"]))
        if shared:
            reasons.append("shared style: " + ", ".join(shared[:2]))

        # audio features where this song is closest to the seeds (in z-scores)
        diff = np.abs(self.audio_z[i] - self.audio_z[seed_rows].mean(axis=0))
        close = [AUDIO_FEATURES[j] for j in np.argsort(diff)
                 if diff[j] < 0.35 and AUDIO_FEATURES[j] in EXPLAIN_FEATURES][:3]
        if close:
            reasons.append("similar " + ", ".join(close))
        return reasons


def split_query(query):
    for sep in [" - ", " by "]:
        if sep in query:
            title, artist = query.split(sep, 1)
            return clean_text(title), clean_text(artist)
    return clean_text(query), ""
