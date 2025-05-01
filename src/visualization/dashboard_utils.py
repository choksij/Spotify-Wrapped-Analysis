import streamlit as st
import pandas as pd
from pathlib import Path

from src.preprocessing.utils import get_project_root


def set_page_config():
    
    st.set_page_config(
        page_title="Spotify Wrapped Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )


@st.cache_data
def load_csv(name: str, index_col=None, parse_dates=None) -> pd.DataFrame:
    
    root = get_project_root()
    path = root / "data" / "processed" / name
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)


def show_key_metrics(df_tracks: pd.DataFrame):
    
    n_tracks = df_tracks["track_id"].nunique()
    n_artists = df_tracks["artist"].nunique()
    n_genres = df_tracks["genre"].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tracks", n_tracks)
    col2.metric("Unique Artists", n_artists)
    col3.metric("Unique Genres", n_genres)


def sidebar_date_filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    start, end = st.sidebar.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    mask = (df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)
    return df.loc[mask]


def select_feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    
    return [c for c in df.columns if c.startswith(prefix)]


def show_dataframe(df: pd.DataFrame, height: int = 400):
    
    st.dataframe(df, height=height, use_container_width=True)
