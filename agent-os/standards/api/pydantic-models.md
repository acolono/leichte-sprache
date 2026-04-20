# Pydantic Models

All API request/response models use Pydantic `BaseModel`.

## Naming Convention

- Field names: **English** (API contract)
- Field descriptions: **German** (for Swagger UI)
- Include `json_schema_extra` with examples on key fields

```python
class ExampleResponse(BaseModel):
    rule_id: str = Field(
        description="ID der Regel die den Verstoß gefunden hat",
        json_schema_extra={"example": "regel_fremdwoerter_issue"},
    )
```

- Use `ConfigDict(str_strip_whitespace=False)` on request models with text input
- Use `Field(...)` for required fields, `Field(default=...)` for optional
