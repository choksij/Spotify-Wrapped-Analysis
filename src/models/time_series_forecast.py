import logging
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

from src.preprocessing.utils import get_project_root, read_json_dir


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def load_play_counts(raw_dir: Path) -> pd.Series:
    
    logger.info("Loading recently played JSON from %s", raw_dir)
    blobs = read_json_dir(raw_dir, pattern="recently_played_*.json")

    
    played = []
    for blob in blobs:
        items = blob.get("items", [])
        for it in items:
           
            played.append(it["played_at"])

    
    df = pd.DataFrame({"played_at": pd.to_datetime(played)})
    df = df.set_index("played_at")
    
    ts = df.resample("W").size()
    ts.index = ts.index.to_period("W").to_timestamp()
    logger.info("Aggregated %d total plays into %d weekly points", len(played), len(ts))
    return ts


def train_and_save():
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "spotify_api"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    
    ts = load_play_counts(raw_dir)

    
    logger.info("Fitting ARIMA(1,1,1) to weekly play counts...")
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()
    logger.info("ARIMA model summary:\n%s", model_fit.summary())

    
    forecast = model_fit.forecast(steps=4)
    logger.info("4-week forecast:\n%s", forecast)

    
    model_file = model_dir / "ts_forecast_model_v1.pkl"
    joblib.dump(model_fit, model_file)
    logger.info("Saved ARIMA model to %s", model_file)

    
    forecast_file = root / "data" / "processed" / "play_counts_forecast.csv"
    forecast.to_csv(forecast_file, header=["forecast"])
    logger.info("Saved forecast to %s", forecast_file)


if __name__ == "__main__":
    train_and_save()
