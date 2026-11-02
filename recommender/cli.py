import argparse

from .engine import Recommender


def print_results(results):
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. {r.name} - {r.artist}")
        line = f"    sound match {r.sound_match:.2f}"
        if r.reasons:
            line += " | " + "; ".join(r.reasons)
        print(line)


def add_options(p):
    p.add_argument("-n", type=int, default=20, help="how many songs")
    p.add_argument("--diversity", type=float, default=0.3, help="0 = closest matches, 1 = most varied")
    p.add_argument("--popularity", type=float, default=0.1, help="boost for popular songs (0-1)")
    p.add_argument("--collab", type=float, default=0.4, help="weight on playlist co-occurrence (0-1)")
    p.add_argument("--max-per-artist", type=int, default=2)
    p.add_argument("--new-artists", action="store_true", help="don't recommend the seed artists")
    p.add_argument("--genre", action="append", help="only these genres: pop rap rock latin r&b edm")
    p.add_argument("--dislike", action="append", default=[], help="song to steer away from")


def get_recs(rec, seeds, args, seed_weights=None):
    return rec.recommend(
        seeds,
        n=args.n,
        dislikes=[rec.find(d) for d in args.dislike],
        seed_weights=seed_weights,
        diversity=args.diversity,
        popularity=args.popularity,
        collab=args.collab,
        max_per_artist=args.max_per_artist,
        new_artists_only=args.new_artists,
        genres=args.genre,
    )


def main():
    parser = argparse.ArgumentParser(prog="recommender")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="search the dataset")
    p.add_argument("query")

    p = sub.add_parser("recommend", help="get recommendations from some songs")
    p.add_argument("songs", nargs="+", help='e.g. "Basket Case - Green Day"')
    add_options(p)

    p = sub.add_parser("spotify", help="use your spotify account for the seeds")
    p.add_argument("--source", choices=["top", "recent", "saved", "playlist"], default="top")
    p.add_argument("--playlist", help="playlist link if --source playlist")
    p.add_argument("--time-range", choices=["short_term", "medium_term", "long_term"], default="medium_term")
    p.add_argument("--save", metavar="NAME", help="save the results as a playlist")
    add_options(p)

    args = parser.parse_args()
    rec = Recommender.from_csv()

    if args.command == "search":
        for tid, row in rec.search(args.query).iterrows():
            print(f"{tid}  {row.track_name} - {row.track_artist}  {row.genres}")

    elif args.command == "recommend":
        seeds = []
        for s in args.songs:
            try:
                tid = rec.find(s)
            except KeyError as e:
                print(e.args[0])
                continue
            print("seed:", rec.songs.loc[tid, "track_name"], "-", rec.songs.loc[tid, "track_artist"])
            seeds.append(tid)
        if seeds:
            print()
            print_results(get_recs(rec, seeds, args))

    elif args.command == "spotify":
        from . import spotify_api

        sp = spotify_api.connect()
        tracks = spotify_api.get_tracks(sp, args.source, args.time_range, args.playlist)
        seeds, missing = spotify_api.match_to_dataset(rec, tracks)
        print(f"found {len(seeds)} of your {len(tracks)} songs in the dataset\n")
        if not seeds:
            return
        # songs higher up your top list count a bit more
        weights = [1 / (1 + 0.05 * i) for i in range(len(seeds))]
        results = get_recs(rec, seeds, args, weights)
        print_results(results)
        if args.save:
            url = spotify_api.save_playlist(sp, args.save, [r.track_id for r in results])
            print("\nsaved playlist:", url)
