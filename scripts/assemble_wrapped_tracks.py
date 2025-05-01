
import pandas as pd
from pathlib import Path


interim = Path("data") / "interim"
files = {
    "top_tracks.csv":           ["id",         "name",         "artists"],
    "recently_played.csv":      ["track.id",   "track.name",   "track.artists"],
    "saved_tracks.csv":         ["track.id",   "track.name",   "track.artists"],
    "playlist_tracks_top3.csv": ["track_id",   "track_name",   "artists"]
}

records = []
for fname, cols in files.items():
    p = interim / fname
    if not p.exists():
        print(f"Missing {p}, skipping.")
        continue

    df = pd.read_csv(p)

    
    mapping = {
        cols[0]: "track_id",
        cols[1]: "track_name",
        cols[2]: "artists"
    }
    df = df.rename(columns=mapping)

    
    def flatten_artists(val):
        if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
            try:
                lst = pd.eval(val)
                if isinstance(lst, list):
                    return ", ".join(lst)
            except Exception:
                pass
        return val

    df["artists"] = df["artists"].apply(flatten_artists)

    records.append(df[["track_id", "track_name", "artists"]])

if not records:
    raise RuntimeError("No interim track files found!")


wrapped = pd.concat(records, ignore_index=True)
wrapped = wrapped.dropna(subset=["track_id"]).drop_duplicates(subset=["track_id"])


out = interim / "wrapped_tracks.csv"
wrapped.to_csv(out, index=False)
print(f"Wrote {len(wrapped)} unique tracks to {out}")
