import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.preprocessing.utils import get_project_root, read_json_dir, save_df_csv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def clean_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()


    audio_cols = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "duration_ms"
    ]
    for col in audio_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    df = df.drop_duplicates(subset=["track_id"])
    return df


def main():
    project_root = get_project_root()


    raw_json_dir = project_root / "data" / "raw" / "spotify_api"
    csv_fallback = project_root / "data" / "raw" / "full_track_pool" / "dataset.csv"


    json_files = list(raw_json_dir.glob("audio_features_*.json"))
    if json_files:
        logger.info("Found %d audio_features JSON files. Loading...", len(json_files))

        records = read_json_dir(raw_json_dir, pattern="audio_features_*.json")
        df_json = pd.DataFrame(records)


        if not df_json.empty and "id" in df_json.columns:
            df_raw = df_json
            logger.info("Using %d rows from JSON source", len(df_raw))
        else:
            logger.warning(
                "JSON source missing 'id' or empty—falling back to CSV: %s",
                csv_fallback
            )
            df_raw = pd.read_csv(csv_fallback)
    else:
        logger.warning(
            "No audio_features JSON found at %s; reading CSV fallback: %s",
            raw_json_dir, csv_fallback
        )
        df_raw = pd.read_csv(csv_fallback)


    if "id" in df_raw.columns:
        df_raw = df_raw.rename(columns={"id": "track_id"})
    elif "track_id" in df_raw.columns:

        pass
    else:
        raise KeyError(
            f"Couldn't find identifier column in raw audio features. "
            f"Available columns: {df_raw.columns.tolist()}"
        )


    df_raw = df_raw.dropna(subset=["track_id"])


    df_clean = clean_audio_features(df_raw)


    out_path = project_root / "data" / "interim" / "audio_features.csv"
    save_df_csv(df_clean, out_path)
    logger.info("Saved cleaned audio features to %s", out_path)


if __name__ == "__main__":
    main()
