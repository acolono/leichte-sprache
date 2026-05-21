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

# Layer 2b: Pre-fetch Hugging Face Hub assets the rules need at runtime.
# Without this layer the container needs outbound access to huggingface.co
# on first request, and an offline runtime trips obscure errors like
# "stat: path should be ... not NoneType" inside huggingface_hub.
#
# Targets:
#   - bert-base-german-cased                       (mehrere_aussagen, personalpronomen)
#       — into HF_HOME cache. Also used to backfill the abkuerzungen and
#       mehrere_aussagen tokenizer assets after the GitHub-release tarball
#       extract (the tarball ships only fine-tuned weights, not vocab.txt /
#       tokenizer.json).
#   - MiriUll/distilbert-german-text-complexity    (komplexitaet)
#       — uses an explicit `cache_dir=regeln/komplexitaet/textkomplexitaet/data`
#       in code, so we mirror that path here instead of relying on HF_HOME.
#   - fefeefef/leichte-sprache-zahlwoerter         (zahlwoerter)
#       — `snapshot_download`ed straight into the rule's local model dir so
#       the rule's tier-1 (local-dir) loader resolves it. This avoids any
#       HF Hub indirection at runtime and keeps every ML rule on the same
#       "weights live under regeln/<rule>/model/" convention.
#
# HF_HOME is fixed to a $HOME-independent path so the cache also works when
# the runtime container starts as a non-root user or with HOME unset
# (a common k8s/podman default).
ENV HF_HOME=/opt/huggingface
RUN mkdir -p /app/regeln/komplexitaet/textkomplexitaet/data /app/regeln/zahlwoerter/model
RUN python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# bert-base-german-cased → HF_HOME cache (also reused by the post-tarball
# tokenizer backfill below).
AutoTokenizer.from_pretrained("bert-base-german-cased")
AutoModel.from_pretrained("bert-base-german-cased")

# MiriUll DistilBERT → explicit cache_dir mirror.
KOMPL_CACHE = "/app/regeln/komplexitaet/textkomplexitaet/data"
AutoTokenizer.from_pretrained(
    "MiriUll/distilbert-german-text-complexity", cache_dir=KOMPL_CACHE
)
AutoModelForSequenceClassification.from_pretrained(
    "MiriUll/distilbert-german-text-complexity", cache_dir=KOMPL_CACHE
)

# zahlwoerter → flat snapshot into the rule's local model dir.
# The rule loader (regeln/zahlwoerter/regel.py, NumberWordsModel._load_model)
# checks `regeln/zahlwoerter/model/` first and only falls back to HF Hub if
# the directory is empty. Materializing the snapshot locally short-circuits
# the fallback and ships the weights as part of the image.
snapshot_download(
    repo_id="fefeefef/leichte-sprache-zahlwoerter",
    local_dir="/app/regeln/zahlwoerter/model",
)
PY
# Make the caches readable for any uid the runtime container ends up running as.
RUN chmod -R a+rX /opt/huggingface /app/regeln/komplexitaet /app/regeln/zahlwoerter

# Layer 3: ML models from GitHub Releases (cached unless manifest changes).
# If MODEL_MANIFEST.json points to a private repo, pass a token via BuildKit secret:
#   DOCKER_BUILDKIT=1 docker build --secret id=github_token,env=GITHUB_TOKEN .
# The secret is mounted only for this RUN — it is NOT stored in an image layer.
COPY MODEL_MANIFEST.json ./
COPY scripts/ ./scripts/
RUN --mount=type=secret,id=github_token,required=false \
    GITHUB_TOKEN="$(cat /run/secrets/github_token 2>/dev/null || true)" \
    python scripts/download_models.py

# Layer 3b: Backfill missing tokenizer assets for `abkuerzungen` and
# `mehrere_aussagen`. The models-v1.1 release tarball ships only the
# fine-tuned weights (.safetensors / .bin); the BertTokenizer's
# `vocab.txt` and `tokenizer.json` are NOT in the tarball and NOT
# git-tracked (`.gitignore` excludes `**/vocab.txt` and
# `**/tokenizer.json`). Without them, `AutoTokenizer.from_pretrained(
# <local model dir>)` raises
# `TypeError: stat: path should be string..., not NoneType`
# because `vocab_file` resolves to None. Both rules are fine-tuned
# from `bert-base-german-cased` (already prefetched into HF_HOME), so
# copy that tokenizer's `vocab.txt` + `tokenizer.json` next to the
# weights. Leave the rules' own `tokenizer_config.json` and
# `special_tokens_map.json` untouched — they are git-tracked.
RUN python - <<'PY'
import shutil
import tempfile
from pathlib import Path
from transformers import AutoTokenizer

targets = [
    Path("/app/regeln/abkuerzungen/model"),
    Path("/app/regeln/mehrere_aussagen/model/models/stage_statement_classifier_best"),
]

with tempfile.TemporaryDirectory() as tmp:
    AutoTokenizer.from_pretrained("bert-base-german-cased").save_pretrained(tmp)
    for dst in targets:
        if not dst.exists():
            print(f"SKIP (no model dir): {dst}")
            continue
        for fn in ("vocab.txt", "tokenizer.json"):
            src = Path(tmp) / fn
            if src.exists():
                shutil.copy(src, dst / fn)
                print(f"Backfilled: {dst / fn}")
PY

# ---- Runtime stage: lean production image ----
FROM python:3.12-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy the Hugging Face cache populated in the builder so the rules can run
# fully offline (no huggingface.co reachability required at request time).
# HF_HOME is fixed to a $HOME-independent path so the cache also works when
# the container runs as a non-root user.
ENV HF_HOME=/opt/huggingface
COPY --from=builder /opt/huggingface /opt/huggingface

# Copy application code
COPY api_main.py analysis_service.py config.py ./
COPY regeln/ ./regeln/
COPY tools/ ./tools/
COPY prompts/ ./prompts/
COPY data/ ./data/

# Overlay downloaded ML model files from builder (after code copy to ensure models win)
COPY --from=builder /app/regeln/ ./regeln/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
