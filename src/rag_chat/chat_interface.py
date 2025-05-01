# src/rag_chat/chat_interface.py
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Spotify Wrapped RAG Chat (Local Embeddings Only)")


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


root = Path(__file__).resolve().parents[2]
index_dir = root / "data" / "processed" / "rag_index"

if not index_dir.exists():
    logger.error("RAG index not found at %s. Run indexer first.", index_dir)
    raise RuntimeError(f"Index directory missing: {index_dir}")


EMBED_MODEL = "all-MiniLM-L6-v2"
logger.info("Loading local SBERT embeddings (%s)", EMBED_MODEL)
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    persist_directory=str(index_dir),
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    answer: str

@app.get("/chat", response_model=ChatResponse)
def chat(q: str = Query(..., description="Your question about your listening history")):
    logger.info("Received chat query: %s", q)
    try:

        docs = retriever.get_relevant_documents(q)
        if not docs:
            return ChatResponse(query=q, answer="No matching history found.")

        snippets = "\n\n".join(f"- {d.page_content}" for d in docs)
        return ChatResponse(query=q, answer=snippets)
    except Exception as e:
        logger.error("Error during retrieval: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_model=ChatResponse)
def chat_post(body: ChatRequest):
    return chat(q=body.query)
