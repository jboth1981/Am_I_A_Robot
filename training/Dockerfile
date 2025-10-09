# Dockerfile for Transformer Training
# This creates a container specifically for training the binary sequence transformer

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for training
COPY transformer_requirements.txt .

# Install transformer training requirements (includes torch, numpy, pandas, etc.)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio \
    && pip install --no-cache-dir -r transformer_requirements.txt

# Copy training source code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app/src:/app/shared_model:/app/backend_app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-m", "src.train_transformer", "--help"]
