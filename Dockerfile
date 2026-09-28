FROM python:3.11-slim

# System deps: FFmpeg + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so pip doesn't pull in CUDA packages
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (excluded by .dockerignore)
COPY . .

EXPOSE 8000

# Default command — overridden per service in docker-compose
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
