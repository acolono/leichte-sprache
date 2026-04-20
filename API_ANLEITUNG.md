# Leichte Sprache REST-API - Anleitung

## 🎉 Erfolgreich implementiert!

Die CLI-Logik wurde erfolgreich in eine FastAPI-basierte REST-API umgewandelt. Die API stellt die gesamte Leichte-Sprache-Analyse über HTTP-Endpunkte zur Verfügung.

## 📁 Neue Dateistruktur

```
/projekt/
├── api_main.py                      # 🎯 FastAPI-Anwendung (NEU!)
├── analysis_service.py              # 🔧 Service-Layer mit Kernlogik (NEU!)
├── api_requirements.txt             # 📦 API-spezifische Dependencies (NEU!)
├── regeln/regel_beispiel_komplexe_woerter.py  # 📋 Beispiel-Regel (NEU!)
├── regeln/                          # 📚 Bestehende 21+ Regel-Module
└── ... (bestehende Dateien)
```

## 🚀 Setup & Installation

### 1. Virtual Environment erstellen
```bash
python3.11 -m venv leichte_sprache_api_venv
source leichte_sprache_api_venv/bin/activate  # macOS/Linux
# ODER für Windows: leichte_sprache_api_venv\Scripts\activate
```

### 2. API-Dependencies installieren
```bash
pip install -r api_requirements.txt
```

### 3. SpaCy deutsches Modell herunterladen
```bash
python -m spacy download de_core_news_lg
```

### 4. API starten
```bash
# Development Server mit Auto-Reload:
python api_main.py

# ODER Production Server:
uvicorn api_main:app --host 0.0.0.0 --port 8000
```

Die API ist dann verfügbar unter: **http://localhost:8000**

## 📖 API-Endpunkte

### 🔍 Hauptendpunkt: POST /analyse

Analysiert einen deutschen Text auf Leichte-Sprache-Konformität.

**Request:**
```json
{
  "text": "Die komplexe Administration evaluiert die Implementation der neuen Strategie."
}
```

**Response (Standard):**
```json
{
  "annotated_text": "Die komplexe[regel_schwierige_woerter_issue: Schwieriges Wort] Administration[regel_beispiel_komplexe_woerter_issue: Verwaltung verwenden]...",
  "statistics": {
    "total_violations": 15,
    "unique_violations": 12,
    "violations_by_rule": {
      "regel_fremdwoerter": 3,
      "regel_satzlaenge": 2,
      "regel_komplexe_woerter": 4
    }
  },
  "issues": [
    {
      "rule_id": "regel_fremdwoerter_issue",
      "text": "Administration",
      "message": "Möglicherweise Fremdwort - durch deutsches Wort ersetzen oder erklären."
    }
  ]
}
```

### 📋 Format-Parameter

**Vollständige Analyse (Standard):**
```bash
POST /analyse
# ODER explizit:
POST /analyse?format=full
```

**Nur annotierter Text:**
```bash
POST /analyse?format=annotated_text
```

Response:
```json
{
  "annotated_text": "Der annotierte Text mit Markierungen..."
}
```

### 🩺 Health Check: GET /health

Überprüft API-Status und Dependencies:

```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "Leichte Sprache API",
  "version": "1.0.0"
}
```

### ℹ️ API-Info: GET /info

Informationen über verfügbare Regeln:

```bash
GET http://localhost:8000/info
```

## 🧪 API testen

### Mit cURL:
```bash
# Vollständige Analyse
curl -X POST "http://localhost:8000/analyse" \
  -H "Content-Type: application/json" \
  -d '{"text": "Die komplexe Verwaltung implementiert neue Strategien."}'

# Nur annotierter Text
curl -X POST "http://localhost:8000/analyse?format=annotated_text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Das ist ein einfacher Test."}'
```

### Mit Python requests:
```python
import requests

url = "http://localhost:8000/analyse"
data = {
    "text": "Die Administration evaluiert die Implementation der Strategie."
}

# Vollständige Analyse
response = requests.post(url, json=data)
result = response.json()

print(f"Violations: {result['statistics']['total_violations']}")
print(f"Annotated: {result['annotated_text']}")

# Nur annotierter Text
response = requests.post(url, json=data, params={"format": "annotated_text"})
result = response.json()
print(f"Annotated only: {result['annotated_text']}")
```

### Interactive API Documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔧 Architektur-Übersicht

### Service-Layer (`analysis_service.py`)
- **Hauptfunktion:** `analyse_text(text: str) -> dict`
- **Verantwortlichkeiten:**
  - spaCy-Modell laden und cachen
  - Dynamisches Laden aller Regel-Module aus `regeln/`
  - Text-Verarbeitung und Regel-Anwendung
  - Ergebnis-Strukturierung und Text-Annotation
  - Fehlerbehandlung

### API-Layer (`api_main.py`)
- **Framework:** FastAPI mit Pydantic-Validierung
- **Endpunkte:** `/analyse`, `/health`, `/info`
- **Features:**
  - Automatische OpenAPI/Swagger-Dokumentation
  - CORS-Unterstützung für Frontend-Integration
  - Format-Parameter für flexible Ausgaben
  - Umfassendes Error Handling

### Regel-System (bestehende `regeln/` Module)
- **Interface:** `pruefe_regel(doc: spacy.tokens.Doc) -> List[str]`
- **Kompatibilität:** Alle bestehenden Regel-Module funktionieren unverändert
- **Erweiterbarkeit:** Neue Regeln werden automatisch erkannt und geladen

## ✅ Erfolgreich umgesetzt

✅ **Service-Layer:** Kernlogik erfolgreich aus CLI extrahiert  
✅ **FastAPI-Integration:** REST-API mit vollständiger Dokumentation  
✅ **Pydantic-Validierung:** Type-Safe Request/Response-Modelle  
✅ **Flexible Ausgabe:** format-Parameter für verschiedene Use Cases  
✅ **Error Handling:** Robuste Fehlerbehandlung auf allen Ebenen  
✅ **Regel-Kompatibilität:** Alle bestehenden 21+ Regeln funktionieren  
✅ **Automatische Dokumentation:** Swagger UI und ReDoc  
✅ **Health Monitoring:** /health und /info Endpunkte  
✅ **Production-Ready:** CORS, Startup/Shutdown Events, Logging  

## 🎯 Verwendung in Production

Für Production-Deployment:

```bash
# Mit Gunicorn (empfohlen)
pip install gunicorn
gunicorn api_main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Oder direkt mit uvicorn
uvicorn api_main:app --host 0.0.0.0 --port 8000 --workers 4
```

Die API ist vollständig funktional und bereit für den Einsatz! 🚀