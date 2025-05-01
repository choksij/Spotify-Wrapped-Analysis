# Spotify Wrapped Analysis

A data-driven end-to-end analysis and dashboard suite that mirrors Spotify’s “Wrapped” experience.  
Fetches your personal Spotify listening data, cleans and enriches it (audio features, lyrical topics), trains genre/popularity models, builds a RAG (retrieval-augmented generation) index for chatbot queries, and visualizes everything in interactive notebooks and Streamlit dashboards.  
<br>

---

<br>

## Table of content

- [Features](#-features)  
- [Project Structure](#-project-structure)  
- [Prerequisites](#️-prerequisites)  
- [Installation & Setup](#-installation--setup)  
- [Configuration](#-configuration)  
- [Data Ingestion](#-data-ingestion)  
- [Exploratory Analysis Notebooks](#-exploratory-analysis-notebooks)  
- [Full Data Pipeline](#-full-data-pipeline)  
- [Dashboards](#-dashboards)  
- [Docker](#-docker)  
- [Smoke Test](#-smoke-test)  
- [Deployment & CI (Optional)](#-deployment--ci-optional)  
- [Contributing](#-contributing)  

<br>

---

<br>

### Features

- **Data Ingestion** via Spotify Web API (top tracks, recently played, saved tracks, playlists, profile).  
- **Preprocessing**:  
  - Clean raw audio features (danceability, energy, etc.)  
  - Merge in lyrics topic annotations  
  - Engineer new audio & lyrical features  
- **Modeling**:  
  - Genre classifier (RandomForest)  
  - Popularity predictor (RandomForest regressor)  
  - Time-series forecasting (ARIMA)  
- **RAG Chatbot**: Build a local Chroma index of your listening history + lyrics for conversational querying via FastAPI + LangChain.  
- **Visualization**:  
  - Jupyter notebooks for EDA  
  - Streamlit dashboards (`demo_app.py`, `wrapped_app.py`)  
- **Containerization**: Docker-Compose to spin up API, ML pipeline, and dashboard services.  

<br>

### Project Structure

spotify-wrapped-analysis/\
├── .vscode/              \
│   ├── settings.json      \
│   ├── extensions.json     \
│   ├── tasks.json           \
│   └── launch.json           \
│
├── README.md                  \
├── .gitignore                     \
├── test.py                     \
├── smoke_test.py                     \
├── devcontainer.json           \
├── docker-compose.yml           \
├── requirements.txt             \
│ \
├── data/  \
│   ├── raw/  \
│   │   ├── spotify_api/     \
│   │   └── dataset5_lyrics/  \
│   ├── interim/               \
│   └── processed/              \
│ \
├── notebooks/                      \
│   ├── 01_api_ingestion.ipynb\
│   ├── 02_wrapped_eda.ipynb\
│   ├── 03_genre_classification_demo.ipynb\
│   ├── 04_popularity_split_demo.ipynb\
│   └── 05_lyrics_theme_analysis.ipynb\
│ \
├── src/            \
│   ├── __init__.py
│   ├── data_ingestion/ \
│   │   ├── spotify_client.py     \
│   │   └── fetch_data.py          \
│   ├── preprocessing/ \
│   │   ├── clean_audio_features.py \
│   │   ├── merge_lyrics_topics.py\
│   │   └── utils.py\
│   ├── features/\
│   │   ├── audio_feature_engineering.py\
│   │   └── lyrical_feature_engineering.py\
│   ├── models/\
│   │   ├── genre_classifier.py\
│   │   ├── popularity_predictor.py\
│   │   └── time_series_forecast.py\
│   ├── rag_chat/\
│   │   ├── indexer.py   \
│   │   └── chat_interface.py \
│   └── visualization/\
│       ├── plots.py   \
│       └── dashboard_utils.py \
│\
├── models/            \
│   ├── genre_classifier_v1.pkl  \
│   ├── popularity_model_v1.pkl\
│   └── ts_forecast_model_v1.pkl\
│\
├── dashboards/    \
│   ├── wrapped_app.py \
│   ├── demo_app.py     \
│   └── requirements.txt \
│\
├── scripts/      \
│   ├── run_data_pipeline.py    \
│   └── run_dashboard.sh         \
│\
├── reports/                 \
│   ├── figures/              \
│   ├── final_report.md        \
│   └── presentation.pptx       \
│\
└── docker/          \
    ├── Dockerfile.api    \
    ├── Dockerfile.ml      \
    └── Dockerfile.dash     \

<br>

### Prerequisites
- **Python 3.10+**  
- **Git** with **Git LFS** (for model `.pkl` artifacts)  
- **Java** (if you ever use the BFG Repo-Cleaner)  
- **Spotify Developer Account** (Client ID/Secret)

<br>

### Installation & Setup

- **Clone & enter**  
   ```bash
   git clone https://github.com/choksij/Spotify-Wrapped-Analysis.git
   cd Spotify-Wrapped-Analysis
   ```
- Create & activate a virtual environment
  ```bash
  Virtual env
  pip install virtualenv
  virtualenv venv
  ```
- Install Python deps
  ```bash
  pip install --upgrade pip
  pip install --no-cache-dir -r requirements.txt
  pip install --no-cache-dir -r dashboards/requirements.txt
  ```
- 
  ```bash
  pip install langchain-community
  ```
- 
  ```bash
  pip install notebook nbformat
  ```

<br>

### Configuration

1. Copy .env.example → .env and fill in your Spotify app credentials:
  ```bash
  SPOTIPY_CLIENT_ID=your_client_id
  SPOTIPY_CLIENT_SECRET=your_client_secret
  SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
  ```

  *or*

  ```bash
  $Env:SPOTIPY_CLIENT_ID     = your_client_id
  $Env:SPOTIPY_CLIENT_SECRET = your_client_secret
  $Env:SPOTIPY_REDIRECT_URI  = "http://localhost:8888/callback"
  ```
2. Ensure .env is git-ignored.

<br>

### Data Ingestion

Fetch your personal Spotify data:

```bash
python src/data_ingestion/fetch_data.py --type top_tracks        --time_range medium_term --limit 50
python src/data_ingestion/fetch_data.py --type recently_played  --limit 50
python src/data_ingestion/fetch_data.py --type saved_tracks     --limit 50
python src/data_ingestion/fetch_data.py --type user_profile
python src/data_ingestion/fetch_data.py --type user_playlists   --limit 20
```

*Outputs go to data/raw/.*

<br>

### Exploratory Analysis Notebooks

Open and run each notebook:

1. notebooks/01_api_ingestion.ipynb

2. notebooks/02_wrapped_eda.ipynb

3. notebooks/03_genre_classification_demo.ipynb

4. notebooks/04_popularity_split_demo.ipynb

5. notebooks/05_lyrics_theme_analysis.ipynb


<br>

### Full Data Pipeline

