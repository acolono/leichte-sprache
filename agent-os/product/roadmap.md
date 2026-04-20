# Product Roadmap

## Phase 1: MVP

- REST API for text analysis (`/analyse` endpoint)
- 18 core rule modules covering:
  - Syntax: sentence length, subordinate clauses, passive voice detection
  - Lexical: foreign words, compound words, short words, complexity
  - Stylistic: negations, idioms, personal pronouns
  - Technical: number words, abbreviations, punctuation, genitive, subjunctive
  - ML-based: perplexity scoring, multiple statement detection, synonyms
- Health and info endpoints
- Swagger/OpenAPI documentation
- Docker deployment support

## Phase 2: Post-Launch

- **LLM-based generation**: `/generate` endpoint for automatic Leichte Sprache text transformation using OpenAI, Anthropic, Mistral, or Ollama
- **Web UI / Frontend**: User-facing web interface for non-technical users to paste, analyze, and transform text directly in the browser
- **Batch processing**: Support for analyzing multiple documents or larger text corpora
- **Reporting**: Exportable compliance reports and analytics on text accessibility
- **Additional enhancements**: Rule refinement based on user feedback, expanded language model support
