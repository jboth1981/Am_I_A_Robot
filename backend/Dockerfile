FROM python:3.11-slim

WORKDIR /app

# System dependencies for psycopg2 and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Optional: install transformer dependencies via build arg
ARG INSTALL_TRANSFORMER_DEPS=false
COPY transformer_requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary python-jose[cryptography] passlib[bcrypt] python-multipart email-validator httpx \
  && if [ "$INSTALL_TRANSFORMER_DEPS" = "true" ]; then \
       pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio \
       && pip install --no-cache-dir -r transformer_requirements.txt ; \
     fi

# Copy backend files
COPY . /app

# Copy shared model SDK
COPY shared_model /app/shared_model

# Set environment variables
ENV PYTHONPATH=/app:/app/shared_model

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
