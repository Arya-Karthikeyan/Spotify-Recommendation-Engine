import numpy as np
import pandas as pd
import scipy.sparse as sp


def normalize_rows(m):
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1))).ravel()
    norms[norms == 0] = 1
    return sp.diags(1 / norms) @ m


class PlaylistSimilarity:
    """'People who added X to a playlist also added Y'.

    Builds a (songs x playlists) matrix. Two songs are similar if they show up
    in the same playlists (cosine similarity of their rows). Big playlists get
    down-weighted by 1/sqrt(size) so they don't take over.

    Most songs are only in 1 or 2 playlists so this is pretty sparse, so I also
    do the same thing at the artist level (artists that share playlists).
    """

    def __init__(self, playlists, songs):
        index_of = {tid: i for i, tid in enumerate(songs.index)}
        playlists = playlists[playlists["track_id"].isin(index_of)]

        playlist_codes, _ = pd.factorize(playlists["playlist_id"])
        rows = playlists["track_id"].map(index_of).values
        sizes = np.bincount(playlist_codes)
        values = 1 / np.sqrt(sizes[playlist_codes])
        shape = (len(songs), len(sizes))
        m = sp.csr_matrix((values, (rows, playlist_codes)), shape=shape)
        self.song_matrix = normalize_rows(m)

        # artist x playlist = sum of that artist's song rows
        self.artist_of_song, artists = pd.factorize(songs["artist_key"])
        song_to_artist = sp.csr_matrix(
            (np.ones(len(songs)), (self.artist_of_song, np.arange(len(songs)))),
            shape=(len(artists), len(songs)),
        )
        self.artist_matrix = normalize_rows(song_to_artist @ m)

    def scores(self, seed_rows, seed_weights):
        """Score every song against the seeds. Returns two arrays scaled to 0-1."""
        song_profile = self.song_matrix[seed_rows].T @ seed_weights
        song_scores = self.song_matrix @ song_profile

        seed_artists = self.artist_of_song[seed_rows]
        artist_profile = self.artist_matrix[seed_artists].T @ seed_weights
        artist_scores = (self.artist_matrix @ artist_profile)[self.artist_of_song]

        return scale(song_scores), scale(artist_scores)

    def best_seed(self, row, seed_rows):
        """Which seed shares the most playlists with this song (or None)."""
        overlap = (self.song_matrix[seed_rows] @ self.song_matrix[row].T).toarray().ravel()
        if overlap.max() <= 0:
            return None
        return seed_rows[overlap.argmax()]


def scale(x):
    x = np.asarray(x).ravel()
    return x / x.max() if x.max() > 0 else x
