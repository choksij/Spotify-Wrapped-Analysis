#!/usr/bin/env python3
"""
Orchestrate the full data pipeline:
  1. Clean raw audio features
  2. Merge lyrics topics
  3. Engineer audio features
  4. Engineer lyrical features
  5. Train genre classifier
  6. Train popularity predictor
  7. Fit time-series forecast
  8. Build RAG index
"""
import sys
from pathlib import Path

# ─── Make src importable ────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import logging
from typing import Optional

# pipeline stages
from src.preprocessing.clean_audio_features import main as clean_audio_main
from src.preprocessing.merge_lyrics_topics import main as merge_topics_main
from src.features.audio_feature_engineering import main as audio_feat_main
from src.features.lyrical_feature_engineering import main as lyrical_feat_main
from src.models.genre_classifier import train_and_save as genre_main
from src.models.popularity_predictor import train_and_save as pop_main
from src.models.time_series_forecast import train_and_save as ts_main
from src.rag_chat.indexer import main as rag_index_main


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_rag_index_step(
    pattern: str = "recently_played_*.json",
    max_docs: Optional[int] = None,
    use_local: bool = True,
) -> None:
    """
    Call indexer with the new signature.
    Toggle `use_local=False` if you prefer OpenAI embeddings and have quota.
    """
    logging.info("8️⃣ Building RAG index")
    rag_index_main(pattern, max_docs, use_local)


def main() -> None:
    setup_logging()
    logging.info("🔄 Starting full data pipeline run")

    logging.info("1️⃣ Cleaning raw audio features")
    clean_audio_main()

    logging.info("2️⃣ Merging lyrics & topic features")
    merge_topics_main()

    logging.info("3️⃣ Engineering audio features")
    audio_feat_main()

    logging.info("4️⃣ Engineering lyrical features")
    lyrical_feat_main()

    logging.info("5️⃣ Training genre classifier")
    genre_main()

    logging.info("6️⃣ Training popularity predictor")
    pop_main()

    logging.info("7️⃣ Fitting time-series forecast")
    ts_main()

    # --- NEW: call wrapper so the args match the refactored indexer ----------
    run_rag_index_step()

    logging.info("✅ Data pipeline completed successfully")


if __name__ == "__main__":
    main()
