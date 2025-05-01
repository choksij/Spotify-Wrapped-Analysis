import json
import pandas as pd
import logging
from pathlib import Path
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory %s", parent)


def read_json_file(path: Path) -> Any:
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json_dir(dir_path: Path, pattern: str = "*.json") -> List[Any]:
    
    data = []
    for path in sorted(dir_path.glob(pattern)):
        try:
            data.append(read_json_file(path))
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e, exc_info=True)
    return data


def save_df_csv(df: pd.DataFrame, out_path: Path, index: bool = False) -> None:
    
    ensure_dir(out_path)
    df.to_csv(out_path, index=index)
    logger.info("Saved DataFrame (%s) to %s", df.shape, out_path)


def read_df_csv(path: Path, **kwargs) -> pd.DataFrame:
    
    return pd.read_csv(path, **kwargs)
