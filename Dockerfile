FROM python:3.11-slim

WORKDIR /app

# Copy dependency definitions first for caching
COPY pyproject.toml poetry.lock ./
COPY service_account.json ./service_account.json

# Install Poetry and project dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Ensure FAISS index and metadata are included
COPY faiss_index.faiss doc_metadata.pkl ./

# Copy the rest of the application code
COPY . /app

# Launch the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]