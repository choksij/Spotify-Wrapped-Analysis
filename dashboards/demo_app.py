import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.visualization.plots import histogram, bar_categories, sunburst_hierarchy
from src.visualization.dashboard_utils import set_page_config
from src.preprocessing.utils import read_df_csv, get_project_root


def main():
    set_page_config()
    st.title("Spotify Wrapped – Demo Modules")

    tabs = st.tabs([
        "Genre Distribution",
        "Popularity Comparison",
        "Lyrics & Themes",
    ])

    root = get_project_root()  

    
    with tabs[0]:
        st.header("Genre Distribution Demo")
        genres_path = root / "data" / "raw" / "genres_v2" / "genres_v2.csv"
        df_genres = read_df_csv(genres_path)

        fig1 = bar_categories(
            df_genres,
            category_col="genre",
            top_n=15,
            title="Top 15 Genres in Dataset"
        )
        st.plotly_chart(fig1, use_container_width=True)

        feature = st.selectbox(
            "Select audio feature",
            options=["energy", "danceability", "valence"]
        )
        selected_genre = st.selectbox(
            "Select genre to inspect",
            options=sorted(df_genres["genre"].unique())[:10]
        )
        df_sub = df_genres[df_genres["genre"] == selected_genre]
        fig2 = histogram(
            df_sub,
            column=feature,
            title=f"{feature.capitalize()} Distribution for {selected_genre}"
        )
        st.plotly_chart(fig2, use_container_width=True)

    
    with tabs[1]:
        st.header("High vs Low Popularity Demo")

        splits_dir = root / "data" / "raw" / "popularity_splits"
        high_path = splits_dir / "high_popularity_spotify_data.csv"
        low_path  = splits_dir / "low_popularity_spotify_data.csv"

        df_high = read_df_csv(high_path)
        df_low  = read_df_csv(low_path)

        st.subheader("Energy Distribution")
        fig3 = histogram(
            df_high,
            column="energy",
            title="High-Popularity Tracks Energy"
        )
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = histogram(
            df_low,
            column="energy",
            title="Low-Popularity Tracks Energy"
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Top Playlist Genres")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**High-Popularity**")
            fig5 = bar_categories(
                df_high,
                category_col="playlist_genre",
                top_n=10,
                title="High‐Popularity: Top Playlist Genres"
            )
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.markdown("**Low-Popularity**")
            fig6 = bar_categories(
                df_low,
                category_col="playlist_genre",
                top_n=10,
                title="Low‐Popularity: Top Playlist Genres"
            )
            st.plotly_chart(fig6, use_container_width=True)

    
    with tabs[2]:
        st.header("Lyrics & Topic Themes Demo")
        lyrics_path = root / "data" / "processed" / "lyrics_features.csv"
        df_lyrics = read_df_csv(lyrics_path)

        topic_cols = [c for c in df_lyrics.columns if c.startswith("topic_")]

        st.subheader("Average Topic Scores Across Songs")
        mean_topics = (
            df_lyrics[topic_cols]
            .mean()
            .reset_index()
            .rename(columns={"index": "topic", 0: "avg_score"})
        )
        fig7 = bar_categories(
            mean_topics,
            category_col="topic",
            value_col="avg_score",
            top_n=len(mean_topics),
            title="Avg Topic Scores"
        )
        st.plotly_chart(fig7, use_container_width=True)

        if "main_topic" in df_lyrics.columns and "genre" in df_lyrics.columns:
            st.subheader("Genre → Main Topic Breakdown")
            fig8 = sunburst_hierarchy(
                df_lyrics,
                path=["genre", "main_topic"],
                values=None,
                title="Genres and Main Topics"
            )
            st.plotly_chart(fig8, use_container_width=True)


if __name__ == "__main__":
    main()
