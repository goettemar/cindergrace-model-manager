# CINDERGRACE Model Manager
# Lightweight Gradio app for managing ComfyUI models on RunPod Network Volumes
FROM python:3.11-slim

# Build args
ARG DEBIAN_FRONTEND=noninteractive

# Environment
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MODELS_ROOT=/workspace/models

# System dependencies (aria2c for fast downloads)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# Create default model directories (will be overwritten by Network Volume)
RUN mkdir -p /workspace/models

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
    CMD curl -f http://localhost:7860/ || exit 1

# Start command - directly with python for better logging
CMD ["python", "app.py"]
