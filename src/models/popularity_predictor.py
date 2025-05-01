import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from src.preprocessing.utils import get_project_root


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    
    logger.info("Loading popularity data from %s", path)
    df = pd.read_csv(path)

    df = df.dropna(subset=["popularity"])
    return df


def preprocess(df: pd.DataFrame):
    
    feature_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_ms", "mode"
    ]

    if "explicit" in df.columns:
        df["explicit_flag"] = df["explicit"].astype(int)
        feature_cols.append("explicit_flag")

    X = df[feature_cols].astype(float)
    y = df["popularity"].astype(float)
    return X, y


def train_and_save():
    root = get_project_root()
    data_path = root / "data" / "raw" / "full_track_pool" / "dataset.csv"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)


    df = load_data(data_path)
    X, y = preprocess(df)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }
    rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rfr, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    logger.info("Starting GridSearch for RandomForestRegressor...")
    gs.fit(X_train, y_train)
    best_reg = gs.best_estimator_
    logger.info("Best params: %s", gs.best_params_)


    y_pred = best_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info("Test RMSE: %.3f", rmse)
    logger.info("Test R2: %.3f", r2)


    out_file = model_dir / "popularity_predictor_v1.pkl"
    joblib.dump(best_reg, out_file)
    logger.info("Saved popularity predictor to %s", out_file)


if __name__ == "__main__":
    train_and_save()
