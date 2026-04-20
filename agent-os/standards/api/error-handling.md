# API Error Handling

Use `HTTPException` with German `detail` messages (bilingual support planned).

## Status Codes

- `400` — Bad input (missing/invalid fields)
- `500` — Server error (analysis failure, unexpected exception)
- `503` — Service unavailable (missing API key, package not installed)

## Error Response Shape

```json
{"detail": "German error message"}
```

## Pattern

```python
raise HTTPException(status_code=503, detail="OpenAI nicht verfügbar: OPENAI_API_KEY nicht gesetzt.")
```

- Catch and re-raise `HTTPException` to avoid wrapping
- Log full tracebacks server-side, return safe messages to client
- Check provider prerequisites (API keys, packages) before processing
