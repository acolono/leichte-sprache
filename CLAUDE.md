# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a German "Leichte Sprache" (Simple Language) text analyzer that uses NLP to check compliance with German accessibility writing rules. The system processes German text and identifies violations of simplicity rules to make content more accessible for people with cognitive disabilities. It also provides LLM-based text generation to automatically transform complex German text into Leichte Sprache.

## Key Architecture Components

### Main Entry Points
- [api_main.py](api_main.py) - **Primary interface** - FastAPI REST endpoint for text analysis and generation
- [analysis_service.py](analysis_service.py) - Core analysis service with dynamic rule loading

### Rule System Architecture
- `regeln/` directory contains 18 rule modules organized as subdirectories
- Each module contains: `regel.py`, `config.py`, `__init__.py`, `README.md`
- Each rule module **must** implement `pruefe_regel(doc: spacy.tokens.Doc) -> List[str]`
- Rules analyze spaCy Doc objects and return list of violation messages
- Rules are **automatically discovered and loaded dynamically** at runtime - no manual registration needed

### Core Processing Pipeline
1. Load German spaCy model (`de_core_news_lg`) - singleton pattern for efficiency
2. Process text into spaCy Doc object
3. Apply all rules dynamically discovered from `regeln/` directory
4. Return annotated text with violations

### REST API Architecture
- **Service Layer** ([analysis_service.py](analysis_service.py)): `LeichteSpracheAnalyzer` class with `analyse_text()` method
- **API Layer** ([api_main.py](api_main.py)): FastAPI with Pydantic validation, CORS, Swagger at `/docs`
- **Endpoints**:
  - `/analyse` (POST) - Analyze text for Leichte Sprache violations
  - `/generate` (POST) - Transform text to Leichte Sprache using LLM (requires API key)
  - `/health` (GET), `/info` (GET) - Status endpoints
- **Format Options**: `format=full` (default) or `format=annotated_text` (compact)

### LLM-based Generation (`/generate` endpoint)
- **Provider support**: OpenAI (default), Ollama (local), Mistral, Anthropic
- **Implementation**: [tools/prompt_optimizer.py](tools/prompt_optimizer.py) - iterative LLM optimization
- **Models**: gpt-4o, gpt-4o-mini (OpenAI); mistral-nemo:12b (Ollama); mistral-large-latest (Mistral); claude-sonnet-4, claude-opus-4 (Anthropic)
- **Environment variables**: `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY` (Ollama runs locally)

### ML/Training Components
- `regeln/perplexity_saetze/perplexity_model.pkl` - NLTK Kneser-Ney n-gram model
- `regeln/abkuerzungen/model/` - BERT NER model for abbreviation detection
- `regeln/mehrere_aussagen/model/` - BERT classifier for statement detection
- `regeln/komplexitaet/textkomplexitaet/` - HuggingFace DistilBERT for complexity
- `regeln/zahlwoerter/` - **BERT Token-Classification for number word detection** (Hugging Face: auto-download or local)

## Development Commands

### Setup with uv (Recommended)
```bash
uv venv && source .venv/bin/activate
uv sync
uv run python -m spacy download de_core_news_lg
```

### Running the API
```bash
# Development server with auto-reload
python api_main.py

# Production server
uvicorn api_main:app --host 0.0.0.0 --port 8000

# Swagger documentation at http://localhost:8000/docs
```

### Docker Deployment
```bash
# Build (requires GITLAB_TOKEN for private packages)
docker build --build-arg GITLAB_TOKEN=<token> -t leichte-sprache-api .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY=<key> leichte-sprache-api
```

### Testing
```bash
# Run all tests
python test-suite/test_runner.py

# Test a specific rule
python test-suite/test_runner.py abkuerzungen

# Verbose output
python test-suite/test_runner.py -v

# List available rules
python test-suite/test_runner.py --list
```

Test file format (`test-suite/<rule_name>/*.txt`):
```
# expected: flag|pass
# description: What this test checks

Text to test goes here.
```

**After modifying a rule**: Run `python test-suite/test_runner.py <rule_name>` to verify changes.

### ML Rule Training
See [docs/ml-training.md](docs/ml-training.md) for the full ML rule training and evaluation CLI reference.

Quick start:
```bash
# List trainable rules
python -m tools.ml train --list

# Train a rule
python -m tools.ml train abkuerzungen --promote

# Evaluate a rule
python -m tools.ml evaluate abkuerzungen --verbose
```

## Rule Development

### Creating New Rules
1. Create directory `regeln/[name]/` with:
   - `regel.py` - Must implement `pruefe_regel(doc: spacy.tokens.Doc) -> List[str]`
   - `config.py` - Configuration settings
   - `__init__.py` - Exports `pruefe_regel`
   - `README.md` - Documentation
2. Add description to `_regel_beschreibungen` dict in [analysis_service.py](analysis_service.py):26
3. Create test files in `test-suite/[name]/`
4. Rule is automatically discovered at runtime

### Rule Categories
- **Syntax**: satzlaenge, nebensaetze, passiv_erkennung
- **Lexical**: fremdwoerter, komposita, kurze_woerter, komplexitaet
- **Stylistic**: negationen, redewendungen, personalpronomen
- **Technical**: zahlwoerter, abkuerzungen, interpunktion, genitiv, konjunktiv
- **ML-based**: perplexity_saetze, mehrere_aussagen, synonyme

## Implementation Notes

### Dynamic Rule Loading
- Rules discovered automatically from `regeln/` subdirectories containing `regel.py`
- Uses `importlib.import_module()` to import `regeln.[name]`
- No manual registration needed

### Package Management
- **pyproject.toml** - Source of truth for dependencies
- **uv.lock** - Locked versions for reproducible installs
- Use `uv add <package>` to add dependencies
- Private packages from `gitlab.oida.group` (requires token)

### Performance
- SpaCy model loaded once (singleton pattern)
- `regel_komposita`: ~840ms (compound word analysis)
- `regel_bert_komplexitaet`: Slower on first run (model loading)

### Dependencies
- **Python 3.11+** required
- **spaCy de_core_news_lg** must be downloaded separately
- **torch + transformers** for BERT-based rules
- **personalpronomen** private package from GitLab registry