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

# Expose port 8080 (GCP Cloud Run standard port)
EXPOSE 8080

# Run FastAPI using uvicorn on port 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]