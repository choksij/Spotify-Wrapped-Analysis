import json
import pandas as pd
import logging
from pathlib import Path
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Returns the root of the project (two levels up from this file).
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    """
    Ensure that the parent directory of `path` exists.
    """
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory %s", parent)


def read_json_file(path: Path) -> Any:
    """
    Load a single JSON file and return its contents.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json_dir(dir_path: Path, pattern: str = "*.json") -> List[Any]:
    """
    Load all JSON files matching `pattern` under `dir_path`.
    Returns a list of the parsed JSON objects.
    """
    data = []
    for path in sorted(dir_path.glob(pattern)):
        try:
            data.append(read_json_file(path))
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e, exc_info=True)
    return data


def save_df_csv(df: pd.DataFrame, out_path: Path, index: bool = False) -> None:
    """
    Save DataFrame to CSV, creating directories as needed.
    """
    ensure_dir(out_path)
    df.to_csv(out_path, index=index)
    logger.info("Saved DataFrame (%s) to %s", df.shape, out_path)


def read_df_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame.
    """
    return pd.read_csv(path, **kwargs)
