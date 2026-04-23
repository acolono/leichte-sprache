# ---- Builder stage: install deps, download models ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools, uv, and git (needed for pip install of
# git+https:// dependencies such as german_compound_splitter).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Layer 1: Python deps (cached unless pyproject.toml/uv.lock change)
# Install CPU-only PyTorch separately to avoid pulling CUDA wheels
COPY pyproject.toml uv.lock ./
# Longer timeout + retries — PyPI reads occasionally stall on large wheels
# (e.g. de_core_news_lg is 568 MB) and short defaults cause flaky builds.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate \
    | grep -v '^torch==' > /tmp/requirements.txt && \
    pip install --no-cache-dir --timeout 300 --retries 5 \
      torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 300 --retries 5 \
      -r /tmp/requirements.txt

# Layer 2: spaCy German model (baked into image)
RUN python -m spacy download de_core_news_lg

# Layer 3: ML models from GitHub Releases (cached unless manifest changes).
# If MODEL_MANIFEST.json points to a private repo, pass a token via BuildKit secret:
#   DOCKER_BUILDKIT=1 docker build --secret id=github_token,env=GITHUB_TOKEN .
# The secret is mounted only for this RUN — it is NOT stored in an image layer.
COPY MODEL_MANIFEST.json ./
COPY scripts/ ./scripts/
RUN --mount=type=secret,id=github_token,required=false \
    GITHUB_TOKEN="$(cat /run/secrets/github_token 2>/dev/null || true)" \
    python scripts/download_models.py

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
