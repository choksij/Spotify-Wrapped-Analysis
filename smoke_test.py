import sys, os

# Insert the 'src' directory at the front of sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)


import src.data_ingestion.spotify_client
import src.preprocessing.clean_audio_features
import src.features.audio_feature_engineering
import src.models.genre_classifier
import src.rag_chat.indexer
import src.visualization.plots
print("✔️ All imports OK")
