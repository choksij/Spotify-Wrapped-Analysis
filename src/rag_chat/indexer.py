import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from src.preprocessing.utils import get_project_root, read_json_dir

# ─── Constants ───────────────────────────────────────────────────────────────
# A small, purely-local SBERT model
DEFAULT_LOCAL_EMBED = "all-MiniLM-L6-v2"

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


def build_recently_played_docs(raw_dir: Path, pattern: str) -> List[Document]:
    """
    Read all JSON files matching `pattern` in raw_dir,
    flatten them into Document objects with metadata.
    """
    blobs = read_json_dir(raw_dir, pattern=pattern)
    docs: List[Document] = []
    for blob in blobs:
        for item in blob.get("items", []):
            played_at = item.get("played_at")
            track = item.get("track", {})
            name = track.get("name", "Unknown")
            artists = ", ".join(a.get("name", "") for a in track.get("artists", []))
            text = f"On {played_at}, you played '{name}' by {artists}."
            metadata: Dict[str, Any] = {
                "played_at": played_at,
                "track_name": name,
                "artists": artists
            }
            docs.append(Document(page_content=text, metadata=metadata))
    logger.info("Built %d Document objects from recently played data", len(docs))
    return docs


def index_documents(
    docs: List[Document],
    persist_directory: Path,
    embedding_model: Union[OpenAIEmbeddings, SentenceTransformerEmbeddings]
) -> None:
    """
    Embed and persist a Chroma vector store from the given documents.
    """
    persist_directory.mkdir(parents=True, exist_ok=True)

    if not docs:
        logger.info("No documents to index (docs is empty). Skipping embedding entirely.")
        return

    logger.info("Creating Chroma vectorstore in %s", persist_directory)
    vect = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(persist_directory)
    )
    vect.persist()
    logger.info("Vector store persisted with %d vectors", len(docs))


def main(pattern: str, max_docs: Optional[int], use_local: bool):
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "spotify_api"
    processed_dir = root / "data" / "processed"
    index_dir = processed_dir / "rag_index"

    # Build all docs matching the pattern
    docs = build_recently_played_docs(raw_dir, pattern)
    if max_docs is not None:
        logger.info("Limiting to first %d documents", max_docs)
        docs = docs[:max_docs]

    # Choose embedding backend
    if use_local:
        logger.info("Using local Sentence-Transformer embeddings (%s)", DEFAULT_LOCAL_EMBED)
        embed = SentenceTransformerEmbeddings(model_name=DEFAULT_LOCAL_EMBED)
    else:
        logger.info("Using OpenAI embeddings (API key from OPENAI_API_KEY)")
        embed = OpenAIEmbeddings()

    if not docs:
        logger.warning("No docs to index; exiting without touching Chroma.")
        return

    index_documents(docs, persist_directory=index_dir, embedding_model=embed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and persist RAG index from Spotify recently-played data"
    )
    parser.add_argument(
        "--pattern",
        default="recently_played_*.json",
        help="glob pattern for JSON files under data/raw/spotify_api"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="If set, only index the first N documents (avoids long runs)"
    )
    parser.add_argument(
        "--local_embeddings",
        action="store_true",
        help="If set, use a local Sentence-Transformer instead of OpenAI"
    )
    args = parser.parse_args()

    main(
        pattern=args.pattern,
        max_docs=args.max_docs,
        use_local=args.local_embeddings
    )
