# Spotify Recommendation Engine

A music recommender I built that suggests songs based on ones you already like. It
uses two things:

1. **How songs sound**: Spotify's audio features (energy, danceability, valence etc.) plus genre and release year
2. **Which songs end up in the same playlists**: a basic collaborative filtering approach

It has a command line tool, a web app (Crate Digger) and it can connect to your
Spotify account to use your top songs and save the results as a playlist.

## Why not just use Spotify's API?

I originally wanted to use Spotify's `/recommendations` and `/audio-features` endpoints,
but Spotify removed them for new apps in November 2024. So the recommending is all
done locally using the [TidyTuesday Spotify Songs dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21)
(~32k rows from 471 playlists, ~25k unique songs after removing duplicates). The
Spotify API is only used for reading your listening history and creating playlists.

The downside is the dataset only goes up to January 2020, so newer songs won't be found.

## How it works

**1. Cleaning the data** (`data.py`)
The csv has one row per song *per playlist*, so I group them into one row per song and
keep all the genres it was tagged with. The same song also shows up multiple times with
different ids (single, album, remaster), so those get merged too.

**2. Song vectors** (`engine.py`)
Each song is turned into a vector:
- 9 audio features, standardised (z-scores) and weighted (energy counts more than liveness for example)
- one-hot genres (6) and subgenres (24)
- release year

Every vector is normalised to length 1 so the dot product of two songs is their cosine similarity.

**3. Your taste profile**
The average of the vectors of the songs you pick. If you mark songs you don't like,
half of their average gets subtracted.

**4. Playlist co-occurrence** (`collab.py`)
A songs × playlists matrix. Songs that appear in the same playlists as your seeds get
a higher score. Most songs are only in one or two playlists, so I also do it at the
artist level which helps a lot.

**5. Final score**
```
match = (1 - collab) * sound_similarity + collab * playlist_score
score = (1 - popularity) * match + popularity * song_popularity
```

**6. Diversity**
If you just take the top 20 you often get 20 really similar songs, sometimes all by
the same artist. So I used Maximal Marginal Relevance (MMR) to re-rank: each pick is
penalised by how similar it is to songs already picked, and there's a limit per artist.

## Setup

```bash
pip install -r requirements.txt
```

The dataset is already in `data/`. If it's missing run `python scripts/download_data.py`.

## Usage

### Web app

```bash
uvicorn recommender.app:app
```
Then go to http://127.0.0.1:8000

Search for songs and add them, then use "More like this" / "Not for me" on the results
to tweak them. "Copy Spotify links" copies all the links which you can paste straight
into a Spotify playlist.

### Command line

```bash
python -m recommender search "basket case"
python -m recommender recommend "Smells Like Teen Spirit - Nirvana" "Basket Case - Green Day"
python -m recommender recommend "HUMBLE." --diversity 0.6 --new-artists -n 10
python -m recommender recommend "Blinding Lights" --genre edm --dislike "Yummy - Justin Bieber"
```

Options:
- `-n` number of songs
- `--diversity` 0 to 1, higher = more varied
- `--collab` 0 to 1, how much playlist co-occurrence counts vs sound (default 0.4)
- `--popularity` 0 to 1, boost for popular songs
- `--max-per-artist` default 2
- `--new-artists` don't include artists you already picked
- `--genre` only pop / rap / rock / latin / r&b / edm
- `--dislike` a song to steer away from

### With your Spotify account

1. Make an app at https://developer.spotify.com/dashboard with the redirect URI `http://127.0.0.1:8888/callback`
2. Set your keys:
   ```bash
   export SPOTIPY_CLIENT_ID=your_id
   export SPOTIPY_CLIENT_SECRET=your_secret
   export SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
   ```
3. Run:
   ```bash
   python -m recommender spotify --source top --time-range short_term
   python -m recommender spotify --source playlist --playlist <link> --save "More like this"
   ```

## Results

To check if it actually works I did a "playlist completion" test (`scripts/evaluate.py`).
I took 150 playlists out of the data, hid half the songs in each, gave the other half as
seeds and checked how many of the hidden songs were in the top 50 (recall@50).

The test playlists are completely removed when building the model. I had to rebuild the
genre labels too, because in this dataset a song's genre comes from the playlists it's
in. My first version didn't do this and the content-only model scored 0.150, which was
way too high because it was basically cheating.

| Model | Recall@50 |
|---|---|
| hybrid, collab 0.2 | 0.114 |
| **hybrid, collab 0.4 (default)** | **0.115** |
| hybrid, collab 0.6 | 0.102 |
| hybrid, collab 0.8 | 0.089 |
| content only (sound + genre + year) | 0.085 |
| audio + year only | 0.048 |
| most popular songs | 0.007 |
| random | 0.003 |

Adding the playlist data makes it about 35% better than just using sound + genre.
Going above 0.4 makes it worse, I think because most songs are in so few playlists
that the co-occurrence scores get noisy.

## Tests

```bash
python -m pytest
```

## Things I'd like to add

- A "connect Spotify" button in the web app instead of only the CLI
- A newer/bigger dataset (this one stops at 2020 and only has 6 genres)
- Learning the feature weights instead of picking them by hand
- Matrix factorisation (ALS) instead of plain co-occurrence

## Project structure

```
recommender/
  data.py          loading + cleaning the csv
  engine.py        the recommender
  collab.py        playlist co-occurrence
  spotify_api.py   spotify login, reading tracks, saving playlists
  cli.py           command line tool
  app.py           FastAPI backend for the web app
  static/index.html
scripts/
  evaluate.py
  download_data.py
tests/
data/spotify_songs.csv
```
