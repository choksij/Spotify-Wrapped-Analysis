import logging
import re
import pandas as pd
from pathlib import Path

from src.preprocessing.utils import (
    get_project_root,
    ensure_dir,
    save_df_csv,
    read_df_csv
)

logger = logging.getLogger(__name__)


def clean_text(s: str) -> str:
    
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_lyrics_topics(raw_dir: Path) -> pd.DataFrame:
    
    path = raw_dir / "tcc_ceds_music.csv"
    df = read_df_csv(path)
    logger.info("Loaded lyrics-topic data (%d rows)", len(df))
    # Standardize column names
    df = df.rename(columns={
        "artist_name": "artist",
        "track_name": "track",
        "release_date": "release_year",
        "topic": "main_topic"
    })
    return df


def merge_with_topics(
    df_base: pd.DataFrame,
    df_topics: pd.DataFrame,
    artist_col: str = "artist",
    track_col: str = "track"
) -> pd.DataFrame:
    

    df_base = df_base.copy()
    df_base["artist_clean"] = df_base[artist_col].astype(str).apply(clean_text)
    df_base["track_clean"] = df_base[track_col].astype(str).apply(clean_text)

    df_topics = df_topics.copy()
    df_topics["artist_clean"] = df_topics["artist"].astype(str).apply(clean_text)
    df_topics["track_clean"] = df_topics["track"].astype(str).apply(clean_text)


    topic_cols = [
        col for col in df_topics.columns
        if col not in {"artist", "track", "artist_clean", "track_clean", "release_year", "genre", "lyrics"}
    ]


    df_merged = pd.merge(
        df_base,
        df_topics[["artist_clean", "track_clean"] + topic_cols],
        on=["artist_clean", "track_clean"],
        how="left"
    )


    df_merged = df_merged.drop(columns=["artist_clean", "track_clean"])


    df_merged[topic_cols] = df_merged[topic_cols].fillna(0)

    logger.info(
        "Merged base (%d rows) with topics; now has %d columns",
        len(df_merged), len(df_merged.columns)
    )
    return df_merged


def main():
    root = get_project_root()
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"


    base_path = interim_dir / "wrapped_tracks.csv"
    if not base_path.exists():
        logger.error("%s not found—run your track-collection step first", base_path)
        return
    df_base = read_df_csv(base_path)


    raw_lyrics_dir = root / "data" / "raw" / "dataset5_lyrics"
    df_topics = load_lyrics_topics(raw_lyrics_dir)


    df_enriched = merge_with_topics(
        df_base,
        df_topics,
        artist_col="artists",
        track_col="track_name"
    )
    out_path = processed_dir / "tracks_with_topics.csv"
    ensure_dir(out_path)
    save_df_csv(df_enriched, out_path)


if __name__ == "__main__":
    main()
