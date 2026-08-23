# Dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# torch CPU-only, prima del resto
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install sentence-transformers==3.0.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download bge-small in build
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

ENV HF_HUB_OFFLINE=1

COPY mcp_server.py .

EXPOSE 8000

CMD ["python", "mcp_server.py"]