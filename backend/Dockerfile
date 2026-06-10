FROM python:3.11-slim

WORKDIR /app

# System dependencies for psycopg2 and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Optional: install transformer inference dependencies via build arg
ARG INSTALL_TRANSFORMER_INFERENCE=false
COPY inference_requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir bcrypt==4.3.0 \
  && pip install --no-cache-dir \
       fastapi==0.119.0 \
       uvicorn==0.37.0 \
       sqlalchemy==2.0.44 \
       psycopg2-binary==2.9.11 \
       python-jose[cryptography]==3.5.0 \
       passlib[bcrypt]==1.7.4 \
       python-multipart==0.0.20 \
       email-validator==2.3.0 \
       httpx==0.28.1 \
       pydantic==2.12.2 \
       slowapi==0.1.9 \
  && if [ "$INSTALL_TRANSFORMER_INFERENCE" = "true" ]; then \
       pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio \
       && pip install --no-cache-dir -r inference_requirements.txt ; \
     fi

# Copy backend files
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app:/app/shared_model

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
