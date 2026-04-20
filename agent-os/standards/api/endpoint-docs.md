# Endpoint Documentation

Every endpoint decorator must include:

```python
@app.post(
    "/path",
    responses={
        200: {"description": "...", "model": ResponseModel},
        400: {"description": "...", "model": ErrorResponse},
    },
    summary="Short German summary",
    description="Detailed German description",
)
```

- `summary`: short (1 line), shown in Swagger sidebar
- `description`: detailed, shown in Swagger expanded view
- `responses`: map all possible status codes with descriptions and models
- All text in German
