# docker/Dockerfile.ml

# 1) Base image
FROM python:3.10-slim

# 2) Working directory
WORKDIR /workspace

# 3) Unbuffered logs
ENV PYTHONUNBUFFERED=1

# 4) Install core dependencies (for preprocessing, modeling, RAG indexing)
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir \
    -r requirements.txt \
    langchain-community \
    sentence-transformers \
    chromadb

# 5) Copy in the rest of your project so the pipeline script can be found
COPY . /workspace

# 6) Expose an inference port (optional; you can add an API later)
EXPOSE 5000

# 7) Default command: run the full data-pipeline
#    (shell form here so VSCode’s Dockerfile linter won’t mistake flags for instructions)
CMD python scripts/run_data_pipeline.py \
    --pattern recently_played_*.json \
    --max_docs 50 \
    --local_embeddings
