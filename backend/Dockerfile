FROM python:3.11-alpine

WORKDIR /app

# Install system dependencies for psycopg2
RUN apk add --no-cache gcc musl-dev postgresql-dev

# Install only essential packages
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary python-jose[cryptography] passlib[bcrypt] python-multipart email-validator httpx

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
