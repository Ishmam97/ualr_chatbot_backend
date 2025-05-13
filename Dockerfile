FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Copy dependency files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main --no-root

# Copy application code and data files
COPY main.py retriever.py llm.py /app/
COPY faiss_index.faiss doc_metadata.pkl /app/

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose port dynamically based on PORT environment variable
ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run application with dynamic port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
