# Violation Messages

All violation messages returned by `pruefe_regel` are in German.

## Format

```
<What's wrong> "<problematic text>". Besser: <suggestion>.
```

Examples:

```python
'Fremdwort "Administration" (Kategorie: verwaltung). Besser: "Verwaltung".'
'Negationspartikel "nicht" macht den Satz schwer verständlich. Besser: Formulieren Sie positiv.'
'Satz 1 ist zu komplex (15 Wörter). Besser: Teilen Sie den Satz auf.'
```

- Always quote the problematic word/phrase with `""`
- Always include a "Besser:" suggestion
- Keep messages actionable — tell users what to do, not just what's wrong
