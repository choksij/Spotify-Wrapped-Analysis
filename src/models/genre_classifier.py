import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from src.preprocessing.utils import get_project_root


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    
    logger.info("Loading genre data from %s", path)
    df = pd.read_csv(path)
    
    df = df.dropna(subset=["genre"])
    return df


def preprocess(df: pd.DataFrame):
    
    feature_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo"
    ]
    X = df[feature_cols].astype(float)
    y = df["genre"].astype(str)
    return X, y


def train_and_save():
    root = get_project_root()
    data_path = root / "data" / "raw" / "genres_v2" / "genres_v2.csv"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    
    df = load_data(data_path)
    X, y = preprocess(df)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

  
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    logger.info("Starting GridSearch for RandomForestClassifier...")
    gs.fit(X_train, y_train)
    best_clf = gs.best_estimator_
    logger.info("Best params: %s", gs.best_params_)


    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.3f", acc)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))


    out_file = model_dir / "genre_classifier_v1.pkl"
    joblib.dump(best_clf, out_file)
    logger.info("Saved genre classifier to %s", out_file)


if __name__ == "__main__":
    train_and_save()
