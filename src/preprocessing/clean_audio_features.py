import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.preprocessing.utils import get_project_root, read_json_dir, save_df_csv

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def clean_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on an audio-features DataFrame:
      - Convert known feature columns to numeric
      - Drop duplicates on track_id
    """
    df = df.copy()

    # List of expected audio-feature columns
    audio_cols = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "duration_ms"
    ]
    for col in audio_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicate rows by track_id
    df = df.drop_duplicates(subset=["track_id"])
    return df


def main():
    project_root = get_project_root()

    # Paths for JSON dumps and CSV fallback
    raw_json_dir = project_root / "data" / "raw" / "spotify_api"
    csv_fallback = project_root / "data" / "raw" / "full_track_pool" / "dataset.csv"

    # 1) Try JSON files first
    json_files = list(raw_json_dir.glob("audio_features_*.json"))
    if json_files:
        logger.info("Found %d audio_features JSON files. Loading...", len(json_files))
        # read_json_dir returns a list of dicts; convert to DataFrame
        records = read_json_dir(raw_json_dir, pattern="audio_features_*.json")
        df_json = pd.DataFrame(records)

        # Use JSON only if it has an 'id' column and isn't empty
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

    # 2) Normalize identifier column to 'track_id'
    if "id" in df_raw.columns:
        df_raw = df_raw.rename(columns={"id": "track_id"})
    elif "track_id" in df_raw.columns:
        # CSV fallback already uses 'track_id'
        pass
    else:
        raise KeyError(
            f"Couldn't find identifier column in raw audio features. "
            f"Available columns: {df_raw.columns.tolist()}"
        )

    # 3) Drop rows without a valid track_id
    df_raw = df_raw.dropna(subset=["track_id"])

    # 4) Clean the audio features
    df_clean = clean_audio_features(df_raw)

    # 5) Save to interim
    out_path = project_root / "data" / "interim" / "audio_features.csv"
    save_df_csv(df_clean, out_path)
    logger.info("Saved cleaned audio features to %s", out_path)


if __name__ == "__main__":
    main()
