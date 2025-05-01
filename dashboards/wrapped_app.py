# dashboards/wrapped_app.py
"""
Personal “Spotify Wrapped” dashboard driven by tracks_with_topics.csv
--------------------------------------------------------------------
Expected columns (based on your sample):

    track_id           track_name        artists
    dating             violence          world/life …
    … topic columns …  main_topic        age
    danceability       loudness          acousticness
    instrumentalness   valence           energy

None of the original timestamp / genre fields are required.
The app auto-detects what is available and hides charts that
do not make sense for your dataset.
"""
from pathlib import Path
import sys
import streamlit as st
import pandas as pd

# ── Make sure “src/” is on PYTHONPATH when launching via `streamlit run` ──
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.visualization.dashboard_utils import (
    set_page_config,
    load_csv,
    show_key_metrics,
    show_dataframe,
)
from src.visualization.plots import (
    histogram,
    bar_categories,
)

# ------------------------------------------------------------------------------
# 1) Page config & title
# ------------------------------------------------------------------------------
set_page_config()
st.title("🎧 Your Spotify Wrapped (topics-only demo)")

# ------------------------------------------------------------------------------
# 2) Load data
# ------------------------------------------------------------------------------
DATA_FILE = "tracks_with_topics.csv"
df = load_csv(DATA_FILE)

# Column buckets --------------------------------------------------------------
numeric_audio_cols = [
    c for c in df.columns
    if c in {"danceability", "loudness", "acousticness",
             "instrumentalness", "valence", "energy"}
]
topic_cols = [c for c in df.columns if c.startswith(("dating", "violence", "world/life", "night/time",
                                                    "shake the audience", "family/gospel", "romantic",
                                                    "communication", "obscene", "music", "movement/places",
                                                    "light/visual perceptions", "family/spiritual",
                                                    "like/girls", "sadness", "feelings"))]

# ------------------------------------------------------------------------------
# 3)  High-level metrics (just counts this time)
# ------------------------------------------------------------------------------
n_tracks   = df["track_id"].nunique() if "track_id" in df.columns else len(df)
n_artists  = df["artists"].nunique()  if "artists"  in df.columns else None
n_topics   = len(topic_cols)

col1, col2, col3 = st.columns(3)
col1.metric("Total tracks",  n_tracks)
if n_artists is not None:
    col2.metric("Unique artists", n_artists)
col3.metric("Topic columns",  n_topics)

st.divider()

# ------------------------------------------------------------------------------
# 4)  Audio-feature explorer (if present)
# ------------------------------------------------------------------------------
if numeric_audio_cols:
    st.header("🎚️  Explore audio features")
    feat = st.selectbox("Choose a feature", options=numeric_audio_cols)
    bins = st.slider("Bins", 10, 100, 30, key="bins_audio")
    st.plotly_chart(
        histogram(df, column=feat, title=f"Distribution of {feat}", nbins=bins),
        use_container_width=True
    )
else:
    st.info("No standard audio-feature columns found.")

# ------------------------------------------------------------------------------
# 5)  Topic strength overview
# ------------------------------------------------------------------------------
if topic_cols:
    st.header("📝 Average topic strengths across all tracks")
    mean_topics = (
        df[topic_cols]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "topic", 0: "avg_score"})
    )
    st.plotly_chart(
        bar_categories(
            mean_topics,
            category_col="topic",
            value_col="avg_score",
            top_n=len(mean_topics),
            title="Average topic score"
        ),
        use_container_width=True
    )

# ------------------------------------------------------------------------------
# 6)  Main-topic counts (if column exists)
# ------------------------------------------------------------------------------
if "main_topic" in df.columns:
    st.header("🔖 Main topic distribution")
    st.plotly_chart(
        bar_categories(
            df, category_col="main_topic", top_n=15,
            title="Top main topics"
        ),
        use_container_width=True
    )

# ------------------------------------------------------------------------------
# 7)  Raw preview
# ------------------------------------------------------------------------------
with st.expander("🔍  Preview first 200 rows"):
    show_dataframe(df.head(200))
