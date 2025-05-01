import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing.utils import (
    get_project_root,
    read_df_csv,
    save_df_csv
)

logger = logging.getLogger(__name__)


def load_audio_features(interim_path: Path) -> pd.DataFrame:
    
    logger.info("Loading audio features from %s", interim_path)
    return read_df_csv(interim_path)


def engineer_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    
    df["dance_x_energy"] = df["danceability"] * df["energy"]
    df["valence_x_dance"] = df["valence"] * df["danceability"]
    
    
    df["duration_min"] = df["duration_ms"] / 60000.0
    
    
    df["tempo_bucket"] = pd.cut(
        df["tempo"],
        bins=[0, 90, 120, np.inf],
        labels=["slow", "mid", "fast"]
    ).astype(str)
    
    logger.info("Engineered %d audio features (added %d new cols)",
                df.shape[1], 4)
    return df


def normalize_numeric(
    df: pd.DataFrame,
    exclude: list = None
) -> pd.DataFrame:
    
    exclude = exclude or []
    df = df.copy()
    
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    to_norm = [c for c in num_cols if c not in exclude]
    
    for col in to_norm:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f"{col}_z"] = (df[col] - mean) / std
        else:
            df[f"{col}_z"] = 0.0
        logger.debug("Normalized %s: mean=%.3f std=%.3f", col, mean, std)
    
    logger.info("Created %d normalized columns", len(to_norm))
    return df


def main():
    root = get_project_root()
    interim = root / "data" / "interim" / "audio_features.csv"
    processed_dir = root / "data" / "processed"
    
    
    df_raw = load_audio_features(interim)
    
    
    df_feat = engineer_audio_features(df_raw)
    
    
    df_norm = normalize_numeric(df_feat, exclude=["track_id"])
    
    
    out_path = processed_dir / "audio_features_engineered.csv"
    save_df_csv(df_norm, out_path)
    logger.info("Saved engineered audio features to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s"
    )
    main()
