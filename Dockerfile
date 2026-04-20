# ---- Builder stage: install deps, download models ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Layer 1: Python deps (cached unless pyproject.toml/uv.lock change)
# Install CPU-only PyTorch separately to avoid pulling CUDA wheels
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate \
    | grep -v '^torch==' > /tmp/requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Layer 2: spaCy German model (baked into image)
RUN python -m spacy download de_core_news_lg

# Layer 3: ML models from GitHub Releases (cached unless manifest changes)
COPY MODEL_MANIFEST.json ./
COPY scripts/ ./scripts/
RUN python scripts/download_models.py

# ---- Runtime stage: lean production image ----
FROM python:3.12-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY api_main.py analysis_service.py config.py ./
COPY regeln/ ./regeln/
COPY tools/ ./tools/
COPY prompts/ ./prompts/
COPY data/ ./data/

# Overlay downloaded ML model files from builder (after code copy to ensure models win)
COPY --from=builder /app/regeln/ ./regeln/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
