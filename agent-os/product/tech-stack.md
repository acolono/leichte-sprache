# Tech Stack

## Backend

- **Python 3.11+** — primary language
- **FastAPI** — REST API framework
- **uvicorn** — ASGI server
- **Pydantic** — request/response validation

## NLP / Machine Learning

- **spaCy** (`de_core_news_lg`) — German language model for text processing
- **BERT / DistilBERT** (HuggingFace Transformers) — abbreviation detection, statement classification, text complexity scoring, number word detection
- **NLTK** — Kneser-Ney n-gram model for perplexity scoring
- **PyTorch** — ML framework powering BERT-based rules

## LLM Providers

- **OpenAI** (gpt-4o, gpt-4o-mini) — primary generation provider
- **Anthropic** (claude-sonnet-4, claude-opus-4) — alternative provider
- **Mistral** (mistral-large-latest) — alternative provider
- **Ollama** (mistral-nemo:12b) — local/self-hosted option

## Package Management

- **uv** — Python package manager and virtual environment tool

## Deployment

- **Docker** — containerized deployment
- **GitLab CI/CD** — continuous integration pipeline

## Frontend

N/A (planned for Phase 2)
