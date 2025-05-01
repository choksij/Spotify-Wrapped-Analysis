FROM python:3.10-slim

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir \
    -r requirements.txt \
    langchain-community \
    sentence-transformers \
    chromadb

COPY . /workspace

EXPOSE 5000

CMD python scripts/run_data_pipeline.py \
    --pattern recently_played_*.json \
    --max_docs 50 \
    --local_embeddings
