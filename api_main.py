"""
FastAPI application for Leichte Sprache text analysis.

REST API endpoint for analyzing texts for Leichte Sprache compliance.
Exposes the service logic via HTTP.
"""

import asyncio
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import config  # noqa: F401 — sets up application logging

logger = logging.getLogger(__name__)

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from analysis_service import analyse_text, get_loaded_rule_count  # noqa: E402

# Generator imports (optional - only if pydantic-ai and LLM provider packages are available)
try:
    from tools.agent_optimizer import (
        AVAILABLE_MODELS,
        LLM_PROVIDER_ANTHROPIC,
        LLM_PROVIDER_GOOGLE,
        LLM_PROVIDER_MISTRAL,
        LLM_PROVIDER_OLLAMA,
        LLM_PROVIDER_OPENAI,
        AgentOptimizer,
        generate_system_prompt_from_rules,  # noqa: F401
        get_available_models,  # noqa: F401
        get_available_providers,  # noqa: F401
        load_rule_prompts,  # noqa: F401
    )

    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False
    LLM_PROVIDER_OPENAI = "openai"
    LLM_PROVIDER_OLLAMA = "ollama"
    LLM_PROVIDER_MISTRAL = "mistral"
    LLM_PROVIDER_ANTHROPIC = "anthropic"
    LLM_PROVIDER_GOOGLE = "google"
    AVAILABLE_MODELS = {}

# Global generator instance (initialized in lifespan)
generator_optimizer: Optional["AgentOptimizer"] = None

# LRU cache for optimizers by provider+model (enables cross-request learning)
_optimizer_cache: OrderedDict[str, "AgentOptimizer"] = OrderedDict()
_OPTIMIZER_CACHE_MAX_SIZE = 5


def get_optimizer(provider: str, model: Optional[str]) -> "AgentOptimizer":
    """
    Get or create an AgentOptimizer for the given provider/model.

    Uses LRU cache (max 5 entries) to enable cross-request learning:
    - Patterns accumulated across requests
    - System prompt improvements persist
    - Validates model name against AVAILABLE_MODELS
    """
    # Validate model name if provided
    if model and HAS_GENERATOR:
        valid_models = AVAILABLE_MODELS.get(provider, [])
        if valid_models and model not in valid_models:
            raise ValueError(
                f"Unknown model '{model}' for provider '{provider}'. "
                f"Available: {valid_models}"
            )

    cache_key = f"{provider}:{model or 'default'}"

    if cache_key in _optimizer_cache:
        # Move to end (most recently used)
        _optimizer_cache.move_to_end(cache_key)
        logger.debug("Optimizer cache hit: %s", cache_key)
        return _optimizer_cache[cache_key]

    # Evict LRU entry if cache is full
    if len(_optimizer_cache) >= _OPTIMIZER_CACHE_MAX_SIZE:
        evicted_key, _ = _optimizer_cache.popitem(last=False)
        logger.info("Evicted optimizer from cache: %s", evicted_key)

    logger.info(
        "Creating new AgentOptimizer (provider=%s, model=%s)",
        provider,
        model or "default",
    )

    _optimizer_cache[cache_key] = AgentOptimizer(
        analyse_func=analyse_text,
        prompts_dir="prompts",
        llm_provider=provider,
        llm_model=model,
    )

    return _optimizer_cache[cache_key]


# Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator_optimizer

    # Startup
    logger.info("Leichte Sprache API gestartet")
    logger.info("Lade Regeln und Modelle...")

    # Warmup: one-time test to load rules and models
    try:
        test_result = analyse_text("Startup Test")
        if "error" not in test_result:
            logger.info("Service erfolgreich initialisiert")
            logger.info("%d Regeln geladen", get_loaded_rule_count())
        else:
            logger.warning("Service-Warnung: %s", test_result["error"])
    except Exception as e:
        logger.error("Service-Initialisierung fehlgeschlagen: %s", e)

    # Initialize generator (optional - requires LLM provider)
    if HAS_GENERATOR:
        try:
            import os

            if os.environ.get("OPENAI_API_KEY"):
                generator_optimizer = AgentOptimizer(
                    analyse_func=analyse_text,  # Direct call, no HTTP
                    prompts_dir="prompts",
                )
                logger.info(
                    "Generator initialisiert (%d Regeln)",
                    len(generator_optimizer.rule_prompts),
                )
            else:
                logger.warning(
                    "Generator nicht verfügbar: OPENAI_API_KEY nicht gesetzt"
                )
        except Exception as e:
            logger.warning("Generator-Initialisierung fehlgeschlagen: %s", e)
    else:
        logger.info("Generator nicht verfügbar: pydantic-ai Package nicht installiert")

    yield

    # Shutdown
    logger.info("Leichte Sprache API beendet")


# Initialize FastAPI app
app = FastAPI(
    title="Leichte Sprache API",
    description="REST-API für die Analyse von deutschen Texten auf Leichte-Sprache-Konformität",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Pydantic models for request/response


class AnalyseRequest(BaseModel):
    """Request-Model für die Textanalyse."""

    model_config = ConfigDict(str_strip_whitespace=False, validate_assignment=True)

    text: str = Field(
        ...,
        description='Der zu analysierende deutsche Text. WICHTIG: Anführungszeichen im Text müssen als \\" escaped werden!',
        min_length=0,
        max_length=50000,
        json_schema_extra={
            "example": 'Die komplexe Administration mit dem "Anderen" Text.'
        },
    )


class ViolationStatistics(BaseModel):
    """Statistiken über gefundene Verstöße."""

    total_violations: int = Field(description="Gesamtanzahl aller gefundenen Verstöße")
    unique_violations: int = Field(
        description="Anzahl eindeutiger Verstöße (dedupliziert)"
    )
    violations_by_rule: Dict[str, int] = Field(
        description="Anzahl Verstöße pro Regel",
        json_schema_extra={"example": {"regel_fremdwoerter": 3, "regel_satzlaenge": 2}},
    )


class Issue(BaseModel):
    """Einzelner gefundener Verstoß."""

    rule_id: str = Field(
        description="ID der Regel die den Verstoß gefunden hat",
        json_schema_extra={"example": "regel_fremdwoerter_issue"},
    )
    text: str = Field(
        description="Der problematische Text/Begriff",
        json_schema_extra={"example": "Administration"},
    )
    message: str = Field(
        description="Beschreibung des Problems mit Verbesserungsvorschlag",
        json_schema_extra={
            "example": "Möglicherweise Fremdwort - durch deutsches Wort ersetzen oder erklären."
        },
    )
    start: Optional[int] = Field(
        None,
        description="Start-Position des Fehlers im Original-Text (Character offset, 0-basiert)",
        json_schema_extra={"example": 4},
    )
    end: Optional[int] = Field(
        None,
        description="End-Position des Fehlers im Original-Text (Character offset, 0-basiert, exklusiv)",
        json_schema_extra={"example": 18},
    )


class AnalyseResponse(BaseModel):
    """Vollständige Response der Textanalyse."""

    annotated_text: str = Field(
        description="Der ursprüngliche Text mit Markierungen für problematische Stellen",
        json_schema_extra={
            "example": "Die komplexe[regel_fremdwoerter_issue: Fremdwort ersetzen] Administration..."
        },
    )
    statistics: ViolationStatistics = Field(
        description="Zusammenfassende Statistiken über alle Verstöße"
    )
    issues: List[Issue] = Field(
        description="Liste aller gefundenen Verstöße mit Details"
    )


class AnnotatedTextResponse(BaseModel):
    """Vereinfachte Response nur mit annotiertem Text."""

    annotated_text: str = Field(
        description="Der ursprüngliche Text mit Markierungen für problematische Stellen"
    )


class ErrorResponse(BaseModel):
    """Fehler-Response."""

    error: str = Field(description="Beschreibung des aufgetretenen Fehlers")


# Generator Models


class GenerateRequest(BaseModel):
    """Request-Model für die Textgenerierung in Leichte Sprache."""

    model_config = ConfigDict(str_strip_whitespace=False, validate_assignment=True)

    text: str = Field(
        ...,
        description="Der zu transformierende deutsche Text",
        min_length=1,
        max_length=10000,
        json_schema_extra={
            "example": "Die 4m breite Feuerwehrzufahrt sowie der Reversierplatz für das Feuerwehrfahrzeug sind in jedem Fall freizuhalten."
        },
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=15,
        description="Maximale Anzahl an Iterationen für die Verbesserung. Mit Escalation-Strategien werden mehr Iterationen empfohlen.",
    )
    target_violations: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Ziel-Anzahl an Verstößen, bei der die Generierung stoppt",
    )
    provider: str = Field(
        default="openai",
        description="LLM-Provider: 'openai' (Standard), 'google' (Gemini), 'ollama' (lokal), 'mistral' oder 'anthropic'",
        json_schema_extra={
            "example": "openai",
            "enum": ["openai", "google", "ollama", "mistral", "anthropic"],
        },
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM-Modell (z.B. 'gpt-5-nano' für OpenAI, 'gemini-3-flash' für Google)",
        json_schema_extra={
            "example": "gpt-5-nano",
        },
    )
    debug: Literal["INFO", "WARN", "DEBUG"] = Field(
        default="INFO",
        description="Log-Level: INFO (Standard), WARN (nur Warnungen), DEBUG (Details)",
    )


class IterationDetail(BaseModel):
    """Details einer einzelnen Iterations-Runde."""

    iteration: int = Field(description="Nummer der Iteration (1-basiert)")
    text: str = Field(description="Generierter Text in dieser Iteration")
    violations: int = Field(description="Anzahl Verstöße nach dieser Iteration")


class GenerateResponse(BaseModel):
    """Response der Leichte-Sprache-Generierung."""

    original: str = Field(description="Original-Eingabetext")
    result: str = Field(
        description="Generierter Text in Leichter Sprache",
        json_schema_extra={
            "example": "Die Zufahrt für die Feuerwehr ist 4 Meter breit. Halten Sie die Zufahrt immer frei."
        },
    )
    success: bool = Field(description="True wenn Ziel-Anzahl Verstöße erreicht wurde")
    iterations: int = Field(description="Anzahl durchgeführter Iterationen")
    final_violations: int = Field(
        description="Anzahl verbleibender Verstöße im Ergebnis"
    )
    final_escalation_level: int = Field(
        default=0,
        description="Endgültige Eskalationsstufe (0=Normal, 1=Fokussiert, 2=Hohe Temperatur, 3=Regel-für-Regel)",
    )
    hix: Optional[float] = Field(
        default=None,
        description="HIX-Score (Hohenheimer Verständlichkeitsindex, 0-20). 0=Fachsprache, 18-20=Leichte Sprache.",
    )
    hix_rating: Optional[str] = Field(
        default=None,
        description="HIX-Bewertungskategorie: expert, complex, readable, plain, simple, easy",
    )
    faithfulness_score: Optional[int] = Field(
        default=None,
        description="Semantische Treue zum Original (1-5). 5=perfekt treu, 4=weitgehend treu (minimale Auslassungen), 3=teilweise treu (wichtige Info fehlt), 2=schwach treu (erhebliche Änderungen), 1=nicht treu. Bewertet durch separaten LLM-Richter.",
        json_schema_extra={"example": 4},
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="Grund für das Ende der Iteration: 'target_reached' (Ziel erreicht), 'stagnation' (keine Verbesserung), 'faithfulness_limit' (Treue-Limit erreicht)",
        json_schema_extra={"example": "target_reached"},
    )
    iterations_detail: List[IterationDetail] = Field(
        description="Details zu jeder Iteration"
    )
    provider: str = Field(
        description="Verwendeter LLM-Provider",
        json_schema_extra={"example": "openai"},
    )
    model: str = Field(
        description="Verwendetes LLM-Modell",
        json_schema_extra={"example": "gpt-4o"},
    )


# API endpoints


@app.post(
    "/analyse",
    # No fixed response_model since it varies by format parameter
    responses={
        200: {
            "description": "Erfolgreiche Analyse des Textes",
            "model": AnalyseResponse,
        },
        400: {"description": "Ungültige Anfrage (leerer Text)", "model": ErrorResponse},
        500: {"description": "Serverfehler bei der Analyse", "model": ErrorResponse},
    },
    summary="Text auf Leichte Sprache analysieren",
    description="""
    Analysiert einen deutschen Text auf Konformität mit Leichte-Sprache-Regeln.

    Der Endpunkt wendet automatisch alle verfügbaren Regeln an und gibt eine
    detaillierte Analyse zurück, einschließlich:
    - Annotiertem Text mit Markierungen problematischer Stellen
    - Statistiken über gefundene Verstöße
    - Detaillierte Liste aller Issues mit Verbesserungsvorschlägen

    **Format-Parameter:**
    - Ohne Parameter oder `format=full`: Vollständige JSON-Response
    - `format=annotated_text`: Nur annotierter Text zurückgeben
    """,
)
async def analyse_text_endpoint(
    request: AnalyseRequest,
    format: Optional[Literal["full", "annotated_text"]] = Query(
        None,
        description="Ausgabeformat: 'full' für vollständige Analyse, 'annotated_text' nur für annotierten Text",
    ),
):
    """
    Analyze a text for Leichte Sprache compliance.

    Args:
        request: AnalyseRequest with the text to analyze
        format: Optional query parameter for output format

    Returns:
        AnalyseResponse or AnnotatedTextResponse depending on format parameter

    Raises:
        HTTPException: On empty text (400) or analysis errors (500)
    """
    # Extended input validation
    if request.text is None:
        raise HTTPException(status_code=400, detail="Text-Feld ist erforderlich")

    # Empty strings are allowed - service can handle them correctly
    # Only explicit None values are rejected

    try:
        # Perform analysis
        result = analyse_text(request.text)

        # Check for service errors
        if "error" in result:
            raise HTTPException(
                status_code=500, detail=f"Analysefehler: {result['error']}"
            )

        # Validate that all expected keys are present
        required_keys = ["annotated_text", "statistics", "issues"]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise HTTPException(
                status_code=500,
                detail=f"Service-Antwort unvollständig. Fehlende Schlüssel: {missing_keys}",
            )

        # Format-specific response
        if format == "annotated_text":
            return AnnotatedTextResponse(annotated_text=result["annotated_text"])

        # Default: full response (format=None or format="full")
        return AnalyseResponse(
            annotated_text=result["annotated_text"],
            statistics=ViolationStatistics(
                total_violations=result["statistics"]["total_violations"],
                unique_violations=result["statistics"]["unique_violations"],
                violations_by_rule=result["statistics"]["violations_by_rule"],
            ),
            issues=[
                Issue(
                    rule_id=issue["rule_id"],
                    text=issue["text"],
                    message=issue["message"],
                    start=issue.get("start"),
                    end=issue.get("end"),
                )
                for issue in result["issues"]
            ],
        )

    except HTTPException:
        # Forward HTTP exceptions
        raise
    except KeyError as e:
        # Specific handling for missing dictionary keys
        raise HTTPException(
            status_code=500,
            detail=f"Service-Antwort fehlerhaft - fehlender Schlüssel: {str(e)}",
        ) from e
    except Exception as e:
        # Unexpected errors with more details
        import traceback

        error_details = traceback.format_exc()
        logger.error("API Error: %s", error_details)
        raise HTTPException(
            status_code=500, detail=f"Unerwarteter Serverfehler: {str(e)}"
        ) from e


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        200: {
            "description": "Erfolgreiche Generierung von Leichter Sprache",
            "model": GenerateResponse,
        },
        400: {"description": "Ungültige Anfrage", "model": ErrorResponse},
        503: {"description": "Generator nicht verfügbar", "model": ErrorResponse},
        500: {
            "description": "Serverfehler bei der Generierung",
            "model": ErrorResponse,
        },
    },
    summary="Text in Leichte Sprache umwandeln",
    description="""
    Wandelt einen deutschen Text automatisch in Leichte Sprache um.

    Der Endpunkt verwendet ein LLM zusammen mit dem Leichte-Sprache-Regelwerk,
    um den Text iterativ zu verbessern, bis die Ziel-Anzahl an Verstößen erreicht ist.

    **Unterstützte Provider:**
    - `openai` (Standard): Erfordert OPENAI_API_KEY Umgebungsvariable
    - `google` (Gemini): Erfordert GOOGLE_API_KEY oder GEMINI_API_KEY Umgebungsvariable
    - `ollama` (lokal): Erfordert lokal laufenden Ollama-Server
    - `mistral`: Erfordert MISTRAL_API_KEY Umgebungsvariable
    - `anthropic`: Erfordert ANTHROPIC_API_KEY Umgebungsvariable

    **Verfügbare Modelle:**
    - OpenAI: `gpt-5-nano` (Standard), `gpt-5-mini`, `gpt-5.2`, `gpt-oss-120b`
    - Google: `gemini-3-flash` (Standard), `gemini-3-pro`
    - Ollama: `mistral-nemo:12b` (Standard), `llama3:8b`, `llama3.1:8b`
    - Mistral: `mistral-medium-latest` (Standard), `mistral-large-latest`, `mistral-small-latest`
    - Anthropic: `claude-sonnet-4-5-20250929` (Standard), `claude-opus-4-6`, `claude-haiku-4-5-20251001`

    **Parameter:**
    - `text`: Der zu transformierende Text (max. 10.000 Zeichen)
    - `max_iterations`: Maximale Verbesserungsrunden (1-15, Standard: 10)
    - `target_violations`: Stoppt bei dieser Anzahl Verstöße (0-10, Standard: 2)
    - `provider`: LLM-Provider ('openai', 'google', 'ollama', 'mistral' oder 'anthropic', Standard: 'openai')
    - `model`: LLM-Modell (optional, verwendet Provider-Standard)

    **Eskalations-Strategien:**
    Wenn keine Verbesserung mehr erzielt wird, eskaliert das System automatisch:
    - Level 0 (Normal): Alle Regeln, Standard-Temperatur
    - Level 1 (Fokussiert): Top 5 priorisierte Verstöße
    - Level 2 (Hohe Temperatur): Kreativere Lösungen (Temp 0.9)
    - Level 3 (Regel-für-Regel): Fokus auf eine Regel-Kategorie

    **Hinweis:** Die Generierung kann je nach Textlänge und Modell einige Sekunden dauern.
    """,
)
async def generate_simple_lang(request: GenerateRequest):
    """
    Generate Leichte Sprache from a German text.

    Args:
        request: GenerateRequest with text and optional parameters

    Returns:
        GenerateResponse with original, result and iteration details

    Raises:
        HTTPException: When generator is unavailable (503) or on errors (500)
    """
    import os

    # Determine provider and model
    provider = request.provider
    model = request.model

    # Validate provider
    if provider not in [
        LLM_PROVIDER_OPENAI,
        LLM_PROVIDER_OLLAMA,
        LLM_PROVIDER_MISTRAL,
        LLM_PROVIDER_ANTHROPIC,
        LLM_PROVIDER_GOOGLE,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Ungültiger Provider: {provider}. Verfügbar: openai, ollama, mistral, anthropic, google",
        )

    # Check prerequisites based on provider
    if provider == LLM_PROVIDER_OPENAI:
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="OpenAI nicht verfügbar: OPENAI_API_KEY nicht gesetzt.",
            )
    elif provider == LLM_PROVIDER_OLLAMA:
        # Check if ollama package is available
        try:
            import ollama as _ollama_check  # noqa: F401
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail="Ollama nicht verfügbar: ollama Package nicht installiert. Run: pip install ollama",
            ) from e
    elif provider == LLM_PROVIDER_MISTRAL:
        if not os.environ.get("MISTRAL_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="Mistral nicht verfügbar: MISTRAL_API_KEY nicht gesetzt.",
            )
        try:
            import mistralai as _mistral_check  # noqa: F401
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail="Mistral nicht verfügbar: mistralai Package nicht installiert. Run: pip install mistralai",
            ) from e
    elif provider == LLM_PROVIDER_ANTHROPIC:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="Anthropic nicht verfügbar: ANTHROPIC_API_KEY nicht gesetzt.",
            )
        try:
            import anthropic as _anthropic_check  # noqa: F401
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail="Anthropic nicht verfügbar: anthropic Package nicht installiert. Run: pip install anthropic",
            ) from e
    elif provider == LLM_PROVIDER_GOOGLE:
        if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get(
            "GEMINI_API_KEY"
        ):
            raise HTTPException(
                status_code=503,
                detail="Google nicht verfügbar: GOOGLE_API_KEY oder GEMINI_API_KEY nicht gesetzt.",
            )

    try:
        import time

        # Configure log levels based on debug parameter
        log_level = getattr(logging, request.debug)
        logging.getLogger("api_main").setLevel(log_level)
        logging.getLogger("tools").setLevel(log_level)

        verbose = request.debug != "WARN"

        logger.info(
            "Generate request: provider=%s, model=%s, max_iterations=%d, target_violations=%d",
            provider,
            model or "default",
            request.max_iterations,
            request.target_violations,
        )

        # Get cached optimizer (singleton pattern for cross-request learning)
        optimizer = get_optimizer(provider, model)

        logger.info(
            "Using optimizer: %s (provider=%s, model=%s)",
            type(optimizer).__name__,
            optimizer.llm_provider,
            optimizer.llm_model,
        )

        # Perform generation
        t_start = time.monotonic()
        result = await asyncio.to_thread(
            optimizer.generate,
            text=request.text,
            max_iterations=request.max_iterations,
            target_violations=request.target_violations,
            verbose=verbose,
        )
        elapsed = time.monotonic() - t_start

        logger.info(
            "Generate finished: success=%s, violations=%d, iterations=%d, "
            "stop_reason=%s, elapsed=%.1fs",
            result["success"],
            result["final_violations"],
            result["total_iterations"],
            result.get("stop_reason", "unknown"),
            elapsed,
        )

        # Trigger Layer 1 learning periodically (after accumulating patterns)
        total_failures = sum(optimizer.failure_stats.values())
        if total_failures >= 50 and total_failures % 50 == 0:
            try:
                optimizer.optimize_rule_prompts(min_failures=10)
                logger.info(
                    "Layer 1 optimization triggered (%d patterns)", total_failures
                )
            except Exception as opt_err:
                logger.warning("Layer 1 optimization failed: %s", opt_err)

        # Build response
        return GenerateResponse(
            original=result["original"],
            result=result["final"],
            success=result["success"],
            iterations=result["total_iterations"],
            final_violations=result["final_violations"],
            final_escalation_level=result.get("final_escalation_level", 0),
            hix=result.get("hix"),
            hix_rating=result.get("hix_rating"),
            faithfulness_score=result.get("faithfulness_score"),
            stop_reason=result.get("stop_reason"),
            iterations_detail=[
                IterationDetail(
                    iteration=it["iteration"],
                    text=it["text"],
                    violations=it["violations"],
                )
                for it in result["iterations"]
            ],
            provider=optimizer.llm_provider,
            model=optimizer.llm_model,
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Provider nicht verfügbar: {str(e)}",
        ) from e
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error("Generator Error: %s", error_details)
        raise HTTPException(
            status_code=500,
            detail=f"Fehler bei der Generierung: {str(e)}",
        ) from e


@app.get(
    "/health",
    response_model=Dict[str, str],
    summary="Health Check",
    description="Überprüft den Status der API und der Dependencies",
)
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        Dictionary with status information
    """
    try:
        # Test service functionality with short text
        test_result = analyse_text("Test.")

        if "error" in test_result:
            return {"status": "unhealthy", "error": test_result["error"]}

        return {
            "status": "healthy",
            "service": "Leichte Sprache API",
            "version": "1.0.0",
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Service-Test fehlgeschlagen: {str(e)}",
        }


@app.get(
    "/help/json-escaping",
    response_model=Dict[str, Any],
    summary="Hilfe für JSON-Escaping",
    description="Zeigt wie Texte mit Anführungszeichen korrekt formatiert werden",
)
async def json_escaping_help():
    """
    Help endpoint for correct JSON formatting of texts with quotation marks.
    """
    return {
        "title": "JSON-Escaping Hilfe",
        "problem": "Anführungszeichen in Texten können JSON-Parsing-Fehler verursachen",
        "solution": "Alle Anführungszeichen im Text müssen escaped werden",
        "examples": {
            "falsch": {
                "text": 'Text mit "Anführungszeichen" - FEHLER!',
                "problem": "Unescaped quotes brechen JSON-Struktur",
            },
            "richtig": {
                "text": 'Text mit \\"Anführungszeichen\\" - KORREKT!',
                "explanation": "Backslash vor Anführungszeichen escapt sie",
            },
            "curl_beispiel": {
                "datei_methode": "curl --data @file.json http://localhost:8000/analyse",
                "inline_methode": 'curl -d \'{"text": "Text mit \\\\"Anführungszeichen\\\\""}\'',
                "empfehlung": "Verwende Datei-Methode für komplexe Texte",
            },
        },
        "swagger_ui_tipp": "In der Swagger UI werden Anführungszeichen automatisch escaped - einfach normal eingeben!",
        "test_endpoints": ["/analyse?format=full", "/analyse?format=annotated_text"],
    }


@app.get(
    "/info",
    response_model=Dict[str, Any],
    summary="API Information",
    description="Informationen über verfügbare Regeln und System-Status",
)
async def api_info():
    """
    Return information about the API and available rules.

    Returns:
        Dictionary with API information
    """
    import os

    try:
        # Test analysis to discover available rules
        test_result = analyse_text("Test für Regel-Discovery.")

        if "error" in test_result:
            available_rules = []
        else:
            # Extract available rules from statistics
            available_rules = list(
                test_result.get("statistics", {}).get("violations_by_rule", {}).keys()
            )

        return {
            "api_name": "Leichte Sprache API",
            "version": "1.0.0",
            "description": "REST-API für die Analyse und Generierung deutscher Texte in Leichter Sprache",
            "endpoints": {
                "/analyse": "POST - Textanalyse auf Leichte-Sprache-Konformität",
                "/generate": "POST - Automatische Umwandlung in Leichte Sprache (benötigt OPENAI_API_KEY)",
                "/health": "GET - Health Check",
                "/info": "GET - API-Informationen",
                "/docs": "GET - Interactive API Documentation (Swagger UI)",
                "/redoc": "GET - Alternative API Documentation",
            },
            "verfuegbare_regeln": available_rules
            if available_rules
            else "Keine Regeln geladen",
            "format_optionen": {
                "full": "Vollständige Analyse mit Statistics und Issues",
                "annotated_text": "Nur annotierter Text",
            },
            "generator": {
                "verfuegbar": HAS_GENERATOR,
                "regeln_geladen": len(generator_optimizer.rule_prompts)
                if generator_optimizer
                else 0,
                "provider": {
                    "openai": {
                        "verfuegbar": os.environ.get("OPENAI_API_KEY") is not None,
                        "modelle": AVAILABLE_MODELS.get(LLM_PROVIDER_OPENAI, []),
                        "standard": "gpt-4o",
                    },
                    "ollama": {
                        "verfuegbar": True,  # Ollama availability checked at runtime
                        "modelle": AVAILABLE_MODELS.get(LLM_PROVIDER_OLLAMA, []),
                        "standard": "mistral-nemo:12b",
                        "hinweis": "Erfordert lokal laufenden Ollama-Server",
                    },
                    "mistral": {
                        "verfuegbar": os.environ.get("MISTRAL_API_KEY") is not None,
                        "modelle": AVAILABLE_MODELS.get(LLM_PROVIDER_MISTRAL, []),
                        "standard": "mistral-medium-latest",
                    },
                    "anthropic": {
                        "verfuegbar": os.environ.get("ANTHROPIC_API_KEY") is not None,
                        "modelle": AVAILABLE_MODELS.get(LLM_PROVIDER_ANTHROPIC, []),
                        "standard": "claude-sonnet-4-5-20250929",
                    },
                    "google": {
                        "verfuegbar": os.environ.get("GOOGLE_API_KEY") is not None
                        or os.environ.get("GEMINI_API_KEY") is not None,
                        "modelle": AVAILABLE_MODELS.get(LLM_PROVIDER_GOOGLE, []),
                        "standard": "gemini-3-flash",
                    },
                },
                "hinweis": "Wählen Sie Provider und Modell im /generate Endpunkt",
            },
        }

    except Exception as e:
        return {
            "api_name": "Leichte Sprache API",
            "version": "1.0.0",
            "error": f"Fehler beim Laden der Regel-Informationen: {str(e)}",
        }


# CORS configuration (optional, for frontend integration)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Production sollte das restriktiver sein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events moved to the lifespan handler above

if __name__ == "__main__":
    # Direct execution for development
    logger.info(
        "Development Server - Für Production verwenden Sie: uvicorn api_main:app"
    )
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload bei Code-Änderungen
        log_level="info",
    )
