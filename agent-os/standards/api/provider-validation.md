# LLM Provider Validation

The `/generate` endpoint validates prerequisites before processing:

1. **Provider name** — must be in the allowed list
2. **API key** — env var must be set (except Ollama)
3. **Package** — Python package must be importable
4. **Model name** — validated against `AVAILABLE_MODELS[provider]`

Validation order: provider → API key → package → model.

Fail with `503` for missing keys/packages, `400` for invalid provider/model.

```python
# Pattern for each provider:
if not os.environ.get("PROVIDER_API_KEY"):
    raise HTTPException(status_code=503, detail="... nicht verfügbar: KEY nicht gesetzt.")
try:
    import provider_package
except ImportError:
    raise HTTPException(status_code=503, detail="... nicht verfügbar: Package nicht installiert.")
```
