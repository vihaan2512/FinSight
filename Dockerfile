FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 10000 (Render default port)
EXPOSE 10000

# Run FastAPI using uvicorn, dynamically binding to Render's $PORT env variable
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]