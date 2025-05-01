import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocessing.utils import (
    get_project_root,
    read_df_csv,
    save_df_csv
)

logger = logging.getLogger(__name__)


def load_lyrics_data(raw_dir: Path) -> pd.DataFrame:
    
    path = raw_dir / "tcc_ceds_music.csv"
    df = read_df_csv(path)
    logger.info("Loaded lyrics dataset (%d rows)", len(df))
    return df


def engineer_lyrical_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    
    df["lyrics"] = df["lyrics"].fillna("").astype(str)
    
    
    df["char_count"] = df["lyrics"].apply(len)
    df["word_count"] = df["lyrics"].apply(lambda x: len(x.split()))
    
    
    df["unique_words"] = df["lyrics"].apply(
        lambda x: len(set(x.split()))
    )
    
    df["lexical_diversity"] = np.where(
        df["word_count"] > 0,
        df["unique_words"] / df["word_count"],
        0.0
    )
    
    
    def avg_word_len(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)
    
    df["avg_word_length"] = df["lyrics"].apply(avg_word_len)
    
    
    if "topic" in df.columns:
        df = pd.concat(
            [df, pd.get_dummies(df["topic"], prefix="topic")],
            axis=1
        )
        logger.info("One-hot encoded %d topic categories", df["topic"].nunique())
    
    logger.info("Engineered lyrical features (total cols=%d)", df.shape[1])
    return df


def main():
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "dataset5_lyrics"
    processed_dir = root / "data" / "processed"
    
    
    df_raw = load_lyrics_data(raw_dir)
    
    
    df_feat = engineer_lyrical_features(df_raw)
    
    
    out_path = processed_dir / "lyrics_features.csv"
    save_df_csv(df_feat, out_path)
    logger.info("Saved lyrical features to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s"
    )
    main()
