FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make src importable
ENV PYTHONPATH=/app

# Cloud Run uses port 8080
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
