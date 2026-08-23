# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install torch==2.4.0 sentence-transformers==3.0.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server.py .

EXPOSE 8000

CMD ["python", "mcp_server.py"]