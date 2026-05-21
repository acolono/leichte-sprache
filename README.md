# Leichte Sprache Rulez

> Deutsche NLP-API zur Prüfung von Leichter Sprache nach 18 Regeln. Mit LLM-Unterstützung für automatische Vereinfachung.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Lizenz](https://img.shields.io/badge/lizenz-TBD-lightgrey.svg)](#lizenz)

Leichte Sprache Rulez prüft deutsche Texte auf Verstöße gegen die Regeln der Leichten Sprache — Wortwahl, Satzlänge, Komposita, Fremdwörter, Passivkonstruktionen und mehr. Die REST-API liefert annotierten Text mit exakten Positionen und Begründungen pro Regel. Optional transformiert der `/generate`-Endpunkt komplexe Texte per LLM in Leichte Sprache.

Zielgruppe: **Behörden, Redaktionen, Bildungseinrichtungen, Content-Teams und Entwickler**, die barrierefreie Sprache automatisiert prüfen oder integrieren wollen.

---

## Überblick

- **18 Regeln** in 5 Kategorien (Syntax, Lexik, Stil, Technik, ML-basiert)
- **4 ML-basierte Regeln** mit PyTorch/BERT-Modellen (Abkürzungen, Mehrfach-Aussagen, Pronomen-Analyse, Perplexität)
- **5 LLM-Provider** für `/generate`: OpenAI, Anthropic, Mistral, Google, Ollama
- **Docker Compose Deployment** — ein Kommando, Healthcheck inklusive
- **Checksummen-verifizierte Modell-Downloads** aus GitHub Releases zur Build-Zeit

**Stack:** FastAPI · spaCy (`de_core_news_lg`) · PyTorch (CPU) · Transformers · NLTK · uv

---

## Schnellstart (3 Kommandos)

Voraussetzung: Docker mit BuildKit, `gh` CLI eingerichtet (`gh auth login`).

```bash
# 1. Klonen und .env vorbereiten
git clone git@github.com:acolono/leichte-sprache.git && cd leichte-sprache \
  && cp .env.example .env \
  && export GITHUB_TOKEN=$(gh auth token)

# 2. Image bauen (BuildKit-Secret, weil ML-Modelle aus privatem Release geladen werden)
DOCKER_BUILDKIT=1 docker build \
  --secret id=github_token,env=GITHUB_TOKEN \
  -t leichte-sprache-rulez-api:latest .

# 3. Starten und Health-Check
docker compose up -d && sleep 90 && curl http://localhost:8000/health
```

Alternativ in einem Schritt (nutzt das in `compose.yaml` deklarierte Build-Secret):

```bash
cp .env.example .env && export GITHUB_TOKEN=$(gh auth token)
docker compose build && docker compose up -d
```

Erwartete Antwort:

```json
{"status":"healthy","service":"Leichte Sprache API","version":"1.0.0"}
```

Test-Aufruf gegen `/analyse`:

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"text":"Der durch die Bundesregierung beschlossene Gesetzesentwurf."}'
```

> **Hinweis zu `GITHUB_TOKEN`:** Der Token wird nur zur Build-Zeit benötigt, weil `scripts/download_models.py` die vier ML-Modelle aus dem privaten Release `models-v1.1` des Repos `acolono/leichte-sprache` lädt. Sobald der Repo öffentlich ist, wird der `--secret` optional — der Dockerfile deklariert `required=false`, der anonyme Pfad funktioniert dann automatisch.

Interaktive API-Dokumentation: **<http://localhost:8000/docs>** (Swagger UI) bzw. **<http://localhost:8000/redoc>**.

Aufräumen: `docker compose down`.

---

## Voraussetzungen

| Komponente | Version / Menge | Zweck |
|------------|-----------------|-------|
| Docker Engine | ≥ 20.10 mit BuildKit | Build und Laufzeit |
| Docker Compose | v2 | Orchestrierung |
| `gh` CLI **oder** Personal Access Token | Scope `contents:read` | Zugriff auf privates Modell-Release |
| Freier Speicher | ~12 GB | Image mit Torch + spaCy + 4 ML-Modellen |
| RAM | ~2 GB | Laufzeitbedarf des Containers |
| Erster Build | ~7–15 min | Wheel-Downloads + spaCy (568 MB) + Modell-Tarball (1 GB) |

---

## Ausführlicher Start (erklärt)

Dieser Abschnitt dröselt die drei Kommandos aus dem Schnellstart auf — für alle, die verstehen wollen, was passiert.

### 1. `.env` anlegen

```bash
cp .env.example .env
```

`.env.example` enthält:

```
MISTRAL_API_KEY=
OLLAMA_HOST=http://localhost:11434
```

Keine dieser Variablen wird für `/analyse` benötigt — der Regel-Pipeline reicht der leere Zustand. Sie werden nur gebraucht, wenn `/generate` gegen den jeweiligen LLM-Provider laufen soll. Weitere Provider-Keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) können ebenfalls in der `.env` gesetzt werden — siehe [Konfiguration](#konfiguration).

### 2. `GITHUB_TOKEN` setzen

Der Container-Build ruft `scripts/download_models.py` auf. Dieses Skript liest `MODEL_MANIFEST.json`, schlägt über die GitHub-API die Asset-URL des Releases `models-v1.1` nach und lädt den 1-GB-Tarball mit vier Modelldateien. Für private Repos benötigt die API eine Bearer-Authentifizierung.

Zwei Wege:

```bash
# Variante A (komfortabel): gh CLI liefert einen kurzlebigen Token
export GITHUB_TOKEN=$(gh auth token)

# Variante B: klassisches Personal Access Token mit contents:read
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

### 3. Build mit BuildKit-Secret

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=github_token,env=GITHUB_TOKEN \
  -t leichte-sprache-rulez-api:latest .
```

> **Hinweis zu `docker build` vs. `docker compose build`:** Die `compose.yaml` deklariert das `github_token`-Secret bereits — `docker compose build` (oder die implizite Build-Phase in `docker compose up`) funktioniert daher äquivalent, solange `$GITHUB_TOKEN` im Shell-Environment gesetzt ist. Der explizite `docker build`-Aufruf bleibt eine valide Alternative; er erzeugt denselben Image-Tag `leichte-sprache-rulez-api:latest`, der in `compose.yaml` via `image:` festgepinnt ist.

Das Dockerfile bindet das Secret gezielt für einen einzelnen `RUN`-Befehl ein:

```dockerfile
RUN --mount=type=secret,id=github_token,required=false \
    GITHUB_TOKEN="$(cat /run/secrets/github_token 2>/dev/null || true)" \
    python scripts/download_models.py
```

**Warum nicht `--build-arg GITHUB_TOKEN=…`?** Build-Args landen in der Image-History und in Zwischenschichten. BuildKit-Secrets nicht — sie existieren nur während der RUN-Instruktion, die sie anfordert, und verlassen das Image nie. `required=false` sorgt dafür, dass der Build auch ohne Secret weiterläuft (fällt dann auf den anonymen Pfad zurück).

### 4. Container starten

```bash
docker compose up -d
```

Compose ruft `docker run` mit Port-Mapping `8000:8000`, Env-File `.env`, Restart-Policy `unless-stopped` und HTTP-Healthcheck auf `/health` auf. Healthcheck läuft alle 30 s mit 60 s Grace-Period beim Start.

### 5. Verifizierung

```bash
docker compose ps                      # STATUS sollte "Up X (healthy)" zeigen
docker compose logs api | grep Regeln  # erwartet: "18 Regeln geladen"
curl http://localhost:8000/health      # erwartet: {"status":"healthy",...}
```

Beispiel-`/analyse`-Antwort für den Test-Satz aus dem Schnellstart (gekürzt):

```json
{
  "annotated_text": "Der durch die Bundesregierung[komposita: …][komplexitaet: …] beschlossene[komposita: …] Gesetzesentwurf[komposita: …].",
  "statistics": {
    "total_violations": 7,
    "unique_violations": 7,
    "violations_by_rule": {"komposita": 3, "komplexitaet": 1, "kurze_woerter": 2, "satzlaenge": 1}
  },
  "issues": [
    {
      "rule_id": "komposita_issue",
      "text": "Bundesregierung",
      "message": "Komplexes Kompositum \"Bundesregierung\" (mittel). Besser: Ersetzen sie durch ein einfacheres wort …",
      "start": 14,
      "end": 29
    }
  ]
}
```

Typische Antwortzeit: **1–2 s beim ersten Aufruf** (ML-Modelle werden lazy geladen), **unter 400 ms warm** auf aktueller Hardware.

---

## Konfiguration

Alle Umgebungsvariablen werden aus `.env` gelesen:

| Variable | Benötigt für | Default | Hinweis |
|----------|--------------|---------|---------|
| `GITHUB_TOKEN` | Docker-**Build** (solange Repo privat) | — | Nur beim `docker build` per `--secret` übergeben — nicht in `.env` stellen. |
| `OPENAI_API_KEY` | `/generate` mit OpenAI-Modellen | — | |
| `MISTRAL_API_KEY` | `/generate` mit Mistral | — | In `.env.example` vorgesehen. |
| `ANTHROPIC_API_KEY` | `/generate` mit Claude | — | |
| `GEMINI_API_KEY` | `/generate` mit Google Gemini | — | |
| `OLLAMA_HOST` | `/generate` mit lokalem Ollama-Server | `http://localhost:11434` | |

Ohne gesetzte LLM-Keys meldet der Container beim Start:

```
WARNING api_main: Generator nicht verfügbar: OPENAI_API_KEY nicht gesetzt
```

`/analyse` läuft davon unabhängig weiter — nur `/generate` liefert dann HTTP 503.

Änderungen an `.env` werden erst nach `docker compose up -d --force-recreate` wirksam. Kein Rebuild nötig.

---

## Endpunkte

| Methode | Pfad | Zweck |
|---------|------|-------|
| `POST` | `/analyse` | Text auf 18 Regeln prüfen. Query-Parameter `format=full` (Standard) oder `format=annotated_text`. |
| `POST` | `/generate` | Text per LLM in Leichte Sprache umwandeln. Body: `text`, optional `provider`, `model`. |
| `GET` | `/health` | Liveness-Probe. |
| `GET` | `/info` | API-Version, Stack-Info, geladene Regeln. |
| `GET` | `/help/json-escaping` | Hilfestellung zur korrekten JSON-Codierung deutscher Sonderzeichen (Umlaute, Anführungszeichen). |
| `GET` | `/docs` | Swagger UI (automatisch von FastAPI). |
| `GET` | `/redoc` | ReDoc (automatisch von FastAPI). |

Vollständige Request-/Response-Schemas: <http://localhost:8000/docs> — dort kann jeder Endpunkt auch direkt ausgeführt werden.

---

## Leichte-Sprache-Regeln (18 Module)

Jede Regel ist ein eigenständiges Modul in `regeln/<name>/` mit `regel.py`, `config.py` und `README.md`. Sie werden beim API-Start automatisch entdeckt und geladen.

### Syntax (3)

- **`satzlaenge`** — Sätze über ~12 Wörtern, durchschnittliche Satzkomplexität
- **`nebensaetze`** — Relativ- und Konjunktionalsätze
- **`passiv_erkennung`** — Passivkonstruktionen

### Lexikalisch (4)

- **`fremdwoerter`** — Anglizismen und lateinische/griechische Bildungen
- **`komposita`** — Komplexe zusammengesetzte Wörter; Vorschläge mit Bindestrichen
- **`kurze_woerter`** — Wörter mit hoher Silbenzahl und Sequenzen langer Wörter
- **`komplexitaet`** — DistilBERT-basierte Wortkomplexitätsprüfung

### Stilistisch (3)

- **`negationen`** — Doppelte Verneinung und negative Formulierungen
- **`redewendungen`** — Idiome und Metaphern
- **`personalpronomen`** — Pronomen-Auflösung per eigenem PyTorch-Multi-Head-Modell

### Technisch (5)

- **`zahlwoerter`** — Zahlen- und Ziffernausdrücke (BERT Token-Classification)
- **`abkuerzungen`** — Abkürzungen und Akronyme (BERT NER)
- **`interpunktion`** — Doppelpunkte, Semikola, Klammersetzung
- **`genitiv`** — Genitivkonstruktionen
- **`konjunktiv`** — Konjunktiv I und II

### ML-basiert (3)

- **`perplexity_saetze`** — NLTK Kneser-Ney n-gram-Modell, bewertet Satzverständlichkeit
- **`mehrere_aussagen`** — BERT-Klassifikator für mehrfach-Aussagen pro Satz
- **`synonyme`** — Synonym-Detektion für einheitliche Terminologie

Die ML-Modelle (4 Gewichts-Dateien, ~1 GB) werden zur Build-Zeit aus `models-v1.1` geladen und ihre SHA256-Summen über `MODEL_MANIFEST.json` geprüft. Lokales Modell-Training ist über `tools/ml` möglich — siehe [`docs/ml-training.md`](docs/ml-training.md).

---

## LLM-Generierung (`/generate`)

Der `/generate`-Endpunkt vereinfacht komplexe deutsche Texte durch iterative LLM-Optimierung gegen die Regel-Pipeline. Unterstützte Provider und Modelle (Stand: aktuelle `tools/agent_optimizer.py`):

| Provider | Modelle |
|----------|---------|
| **OpenAI** | `gpt-5-nano`, `gpt-5-mini` (Standard), `gpt-5.2`, `gpt-oss-120b` |
| **Anthropic** | `claude-sonnet-4-5-20250929` (Standard), `claude-opus-4-6`, `claude-haiku-4-5-20251001` |
| **Mistral** | `mistral-medium-latest` (Standard), `mistral-large-latest`, `mistral-small-latest` |
| **Google** | `gemini-3-flash` (Standard), `gemini-3-pro` |
| **Ollama** | `mistral-nemo:12b` (Standard), `llama3:8b`, `llama3.1:8b` — lokaler Server |

Minimaler Aufruf (Mistral):

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Der durch die Bundesregierung beschlossene Gesetzesentwurf beinhaltet umfangreiche Anpassungen.",
    "provider": "mistral"
  }'
```

Die Antwort enthält den vereinfachten Text, die Anzahl Optimierungs-Iterationen, die finale Verstoßzahl und einen Treuescore (`faithfulness_score`) gegen den Ursprungstext.

### Ollama lokal (zwei Varianten)

**Variante A — In-Network via Compose-Profil (empfohlen):**

`compose.yaml` enthält ein optionales `ollama`-Profil mit einem Ollama-Server **im Compose-Netzwerk** plus einem One-Shot-Warmer, der das Default-Modell (`OLLAMA_DEFAULT_MODEL`, Standard `mistral-nemo:12b`) beim ersten Hochfahren zieht und im Named-Volume `ollama-data` persistiert.

```bash
# Compose bringt api + ollama + ollama-pull zusammen hoch.
# Modell-Pull läuft beim ersten Start ~3–7 min (~7 GB).
docker compose --profile ollama up -d

# Anschließend Generierung gegen das lokale Modell:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"…","provider":"ollama"}'
```

`compose.yaml` überschreibt `OLLAMA_HOST`/`OLLAMA_BASE_URL` automatisch auf `http://ollama:11434` — der Wert in `.env` ist nur für Bare-Metal-Läufe relevant. Der Ollama-Port wird **nicht** auf den Host gemappt, um Konflikte mit einem evtl. lokal installierten Ollama zu vermeiden. Sauberes Aufräumen inkl. heruntergeladener Modelle: `docker compose --profile ollama down -v`.

Disk-Bedarf mit aktivem Profil: ~12 GB Image + ~7 GB Modell = **~19 GB**.

**Variante B — Host-Ollama:**

Wer bereits eine Ollama-Installation auf dem Host betreibt (`brew install ollama` o. Ä.), kann den Container darauf zeigen:

```bash
ollama run mistral-nemo:12b           # einmal pullen + warm halten
# in .env:
OLLAMA_HOST=http://host.docker.internal:11434
docker compose up -d --force-recreate
```

Beide Varianten erfordern **keinen** API-Key.

---

## Entwicklung ohne Docker

Für Beitragende, die auf dem eigenen Rechner arbeiten wollen:

```bash
# 1. Virtuelles Environment und Dependencies
uv venv && source .venv/bin/activate
uv sync

# 2. spaCy-Modell (~568 MB)
uv run python -m spacy download de_core_news_lg

# 3. ML-Modelle laden (GITHUB_TOKEN vorher setzen)
export GITHUB_TOKEN=$(gh auth token)
python scripts/download_models.py

# 4. API starten
python api_main.py        # Auto-Reload, Port 8000

# Alternativ produktionsnah:
uvicorn api_main:app --host 0.0.0.0 --port 8000
```

Voraussetzungen: **Python ≥ 3.11** und [`uv`](https://docs.astral.sh/uv/). Weitere Infos zum ML-Rule-Training siehe [`docs/ml-training.md`](docs/ml-training.md).

### Tests

```bash
# Regel-basierte Tests (203 Tests)
python test-suite/test_runner.py

# Eine einzelne Regel
python test-suite/test_runner.py abkuerzungen -v

# pytest (Unit + Integration)
uv run pytest
```

Test-Dateien in `test-suite/<regel>/*.txt` folgen dem Format:

```
# expected: flag|pass
# description: Was dieser Test prüft

Hier kommt der Prüftext.
```

---

## Fehlerbehebung

**`docker build` schlägt auf `scripts/download_models.py` mit HTTP 404 fehl.**
Ursache: `GITHUB_TOKEN` ist nicht gesetzt oder das gewählte Token hat keinen `contents:read`-Scope. Lösung:

```bash
export GITHUB_TOKEN=$(gh auth token)
gh auth status   # Bestätigt die Scopes
DOCKER_BUILDKIT=1 docker build --secret id=github_token,env=GITHUB_TOKEN -t leichte-sprache-rulez-api:latest .
```

**Container ist `healthy`, aber `/analyse` hängt auf macOS.**
Fast immer ein lokales Proxy-Problem. Zwei typische Ursachen:

1. **System-PAC-URL kollidiert mit Port 8000.** Prüfen mit `scutil --proxy` — wenn `ProxyAutoConfigURLString` auf `http://127.0.0.1:8000/...` zeigt, greift die System-Proxy-Auto-Configuration in die eigene API ein. Lösung: PAC-URL umstellen (`sudo networksetup -setautoproxyurl Wi-Fi <neue-url>`) oder Port-Mapping in `compose.yaml` auf z. B. `8001:8000` ändern.
2. **Cloudflare WARP / corporate Zero-Trust-Client** leitet `localhost`-Traffic um. Erkennbar an Client-IPs aus Cloudflare-Netzen (z. B. `172.65.x.x`) in den Container-Logs und `Invalid HTTP request received`-Warnungen. Lösung: `warp-cli disconnect` für die Session oder Split-Tunnel-Konfiguration.

Verifikation, dass die App selbst korrekt antwortet:

```bash
docker exec leichte-sprache-rulez-api-1 curl -s -X POST \
  http://127.0.0.1:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"text":"Das ist ein einfacher Test."}'
```

**Startup-Log meldet `1 Regeln verfügbar` statt `18 Regeln geladen`.**
Ursache: veraltetes Image mit dem Phase-21-Fehler (`.dockerignore` entfernte Modell-Konfigs aus dem Build-Context). Lösung: Image neu bauen mit aktuellem Quellcode (`DOCKER_BUILDKIT=1 docker build --no-cache --secret id=github_token,env=GITHUB_TOKEN -t leichte-sprache-rulez-api:latest .`).

**`/generate` liefert HTTP 503 — "Generator nicht verfügbar".**
Ursache: Kein LLM-Provider-Key gesetzt. Mindestens eine der Variablen aus [Konfiguration](#konfiguration) in `.env` hinterlegen und `docker compose up -d --force-recreate` ausführen.

---

## Projektstruktur

```
leichte-sprache-rulez/
├── api_main.py                  # FastAPI-App (Endpunkte, Lifespan, Pydantic-Schemas)
├── analysis_service.py          # Kern: LeichteSpracheAnalyzer mit dynamischer Regel-Discovery
├── config.py                    # Logging-Konfiguration
├── regeln/                      # 18 Regel-Module (je regel.py, config.py, __init__.py, README.md)
│   ├── abkuerzungen/ …
│   └── zahlwoerter/ …
├── tools/
│   ├── agent_optimizer.py       # LLM-Provider-Abstraktion, iterative /generate-Pipeline
│   ├── prompt_optimizer.py
│   └── ml/                      # Training/Evaluation CLI (python -m tools.ml …)
├── scripts/
│   ├── download_models.py       # Build-Zeit-Fetch der 4 ML-Modelle aus GitHub Releases
│   └── upload_release.sh        # Maintainer-Helper zum Erstellen des Release-Tarballs
├── prompts/                     # System-Prompts für /generate
├── data/                        # Wörterbücher, Korpora, Referenzdaten
├── test-suite/                  # Regel-basierte .txt-Tests + test_runner.py
├── docs/
│   └── ml-training.md           # ML-Rule-Training Referenz
├── MODEL_MANIFEST.json          # SHA256-Prüfsummen aller ML-Modelle
├── Dockerfile                   # Multi-Stage Build mit BuildKit-Secret
├── compose.yaml                 # Docker-Compose-Service
├── .env.example                 # Template für Umgebungsvariablen
└── pyproject.toml               # Projekt-Metadaten, Dependencies, uv.lock-Basis
```

---

## Tests & CI

**Lokal:**

```bash
python test-suite/test_runner.py    # 203 Regel-basierte Tests
uv run pytest                        # Unit- und Integrations-Tests
```

**CI:** Eine interne GitLab-Pipeline existiert (siehe `CI_CD_PIPELINE.md`). Eine GitHub-Actions-Pipeline ist für ein kommendes Milestone vorgesehen (POL-01).

---

## Mitwirken

Issues und Pull Requests sind willkommen. Ein `CONTRIBUTING.md` mit Details zum Entwicklungs-Workflow, Coding-Standards und Test-Anforderungen folgt.

Für interne Architektur-Hinweise und Coding-Regeln der Beitragenden siehe [`CLAUDE.md`](CLAUDE.md) (enthält Konventionen für neue Regel-Module).

---

## Lizenz

Die Lizenz ist noch festzulegen. Bis dahin bitte vor produktivem Einsatz die Maintainer kontaktieren.
