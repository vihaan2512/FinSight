FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
               from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
               SentenceTransformer('BAAI/bge-large-en-v1.5'); \
               CrossEncoder('BAAI/bge-reranker-large'); \
               AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
               AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]