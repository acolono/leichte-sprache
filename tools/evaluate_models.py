"""
Model Evaluation Script for Leichte Sprache Generation

Systematically tests LLM models across all providers against texts of varying
length and complexity by calling the /generate API endpoint, producing
structured JSONL results and a CSV summary.

All providers now use the agentic tool-calling workflow (AgentOptimizer).

Requires the API server to be running (python api_main.py).

Usage:
    python tools/evaluate_models.py --dry-run
    python tools/evaluate_models.py --providers mistral openai --dry-run
    python tools/evaluate_models.py --models gpt-5-nano mistral-medium-latest
    python tools/evaluate_models.py --lengths short --dry-run
    python tools/evaluate_models.py --resume results/eval_20260214_153042.jsonl
"""

import argparse
import csv
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_PROVIDER = "mistral"

ALL_PROVIDERS = ["openai", "ollama", "mistral", "anthropic", "google"]

MODELS_BY_PROVIDER: Dict[str, List[str]] = {
    "openai": ["gpt-5-nano", "gpt-5-mini", "gpt-5.2", "gpt-oss-120b"],
    "ollama": ["mistral-nemo:12b"],
    "mistral": [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
    ],
    "anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "google": ["gemini-3-flash", "gemini-3-pro"],
}

MODEL_SIZE_TIERS: Dict[str, str] = {
    # Mistral
    "mistral-large-latest": "large",
    "mistral-medium-latest": "medium",
    "mistral-small-latest": "small",
    # OpenAI
    "gpt-5-nano": "small",
    "gpt-5-mini": "medium",
    "gpt-5.2": "large",
    "gpt-oss-120b": "large",
    # Anthropic
    "claude-opus-4-6": "large",
    "claude-sonnet-4-5-20250929": "medium",
    "claude-haiku-4-5-20251001": "small",
    # Ollama
    "mistral-nemo:12b": "medium",
    # Google
    "gemini-3-flash": "small",
    "gemini-3-pro": "large",
}

MODEL_TO_PROVIDER: Dict[str, str] = {
    model: provider
    for provider, models in MODELS_BY_PROVIDER.items()
    for model in models
}

# Per-request timeout — generation can take minutes for long texts
HTTP_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalText:
    text_id: str
    text: str
    length_bin: str
    word_count: int
    category: str


@dataclass
class EvalResult:
    provider: str
    model: str
    model_size_tier: str
    text_id: str
    text_length_bin: str
    text_word_count: int
    success: bool
    final_violations: int
    total_iterations: int
    final_escalation_level: int
    elapsed_seconds: float
    violations_by_rule: Dict[str, int]
    faithfulness_score: Optional[int]
    hix: Optional[float]
    stop_reason: Optional[str]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_size_tier": self.model_size_tier,
            "text_id": self.text_id,
            "text_length_bin": self.text_length_bin,
            "text_word_count": self.text_word_count,
            "success": self.success,
            "final_violations": self.final_violations,
            "total_iterations": self.total_iterations,
            "final_escalation_level": self.final_escalation_level,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "violations_by_rule": self.violations_by_rule,
            "faithfulness_score": self.faithfulness_score,
            "hix": self.hix,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

FALLBACK_CORPUS: List[Dict[str, str]] = [
    {
        "text_id": "short_verwaltung_fb",
        "length_bin": "short",
        "category": "verwaltung",
        "text": (
            "Gemäß der Verwaltungsvorschrift ist die Antragstellung unter "
            "Berücksichtigung sämtlicher Dokumentationspflichten fristgerecht einzureichen."
        ),
    },
    {
        "text_id": "short_technik_fb",
        "length_bin": "short",
        "category": "technik",
        "text": (
            "Die Implementation der systemischen Infrastrukturoptimierung erfordert "
            "eine umfassende Evaluierung der prozessualen Rahmenbedingungen."
        ),
    },
    {
        "text_id": "medium_verwaltung_fb",
        "length_bin": "medium",
        "category": "verwaltung",
        "text": (
            "Die 4m breite Feuerwehrzufahrt sowie der Reversierplatz für das "
            "Feuerwehrfahrzeug sind in jedem Fall freizuhalten. Es darf auch kurzfristig "
            "keine Ware in diesem Bereich gelagert werden."
        ),
    },
    {
        "text_id": "medium_technik_fb",
        "length_bin": "medium",
        "category": "technik",
        "text": (
            "Die Konfiguration des Netzwerk-Routers erfolgt über das webbasierte "
            "Administrationsinterface. Zunächst muss die IP-Adresse des Geräts im "
            "Browser eingegeben werden. Nach der Authentifizierung können die "
            "WLAN-Einstellungen angepasst werden."
        ),
    },
    {
        "text_id": "long_verwaltung_fb",
        "length_bin": "long",
        "category": "verwaltung",
        "text": (
            "Die Beantragung eines Wohnberechtigungsscheins setzt voraus, dass der "
            "Antragsteller die Einkommensgrenzen des Wohnraumförderungsgesetzes nicht "
            "überschreitet. Zur Prüfung der Anspruchsberechtigung sind sämtliche "
            "Einkommensnachweise der letzten zwölf Monate vorzulegen, einschließlich "
            "Lohnabrechnungen, Rentenbescheide und gegebenenfalls Bescheide über den "
            "Bezug von Sozialleistungen. Der Antrag kann persönlich bei der zuständigen "
            "Wohnungsbauförderungsstelle eingereicht werden. Alternativ besteht die "
            "Möglichkeit, die erforderlichen Unterlagen auf dem Postweg einzusenden."
        ),
    },
    {
        "text_id": "long_technik_fb",
        "length_bin": "long",
        "category": "technik",
        "text": (
            "Die Installation einer Photovoltaikanlage auf dem Eigenheim erfordert "
            "zunächst eine Prüfung der statischen Eignung der Dachkonstruktion durch "
            "einen Sachverständigen. Anschließend wird die optimale Ausrichtung und "
            "Dimensionierung der Solarmodule ermittelt, wobei die Dachneigung, die "
            "Himmelsrichtung und mögliche Verschattungen berücksichtigt werden müssen. "
            "Für die Einspeisung des erzeugten Stroms in das öffentliche Netz ist eine "
            "Anmeldung beim zuständigen Netzbetreiber erforderlich."
        ),
    },
]


def load_corpus(corpus_dir: Path) -> List[EvalText]:
    """Parse .txt files from corpus_dir or use fallback corpus."""
    texts: List[EvalText] = []

    if corpus_dir.is_dir():
        for txt_file in sorted(corpus_dir.glob("*.txt")):
            try:
                content = txt_file.read_text(encoding="utf-8")
                meta, body = _parse_corpus_file(content)
                if not body.strip():
                    continue
                text_id = txt_file.stem
                wc = len(body.split())
                texts.append(
                    EvalText(
                        text_id=text_id,
                        text=body.strip(),
                        length_bin=meta.get("length", _infer_length_bin(wc)),
                        word_count=wc,
                        category=meta.get("category", "unknown"),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping %s: %s", txt_file, exc)

    if texts:
        return texts

    # Fallback
    logger.warning("Corpus dir %s empty or missing — using fallback corpus", corpus_dir)
    for item in FALLBACK_CORPUS:
        wc = len(item["text"].split())
        texts.append(
            EvalText(
                text_id=item["text_id"],
                text=item["text"],
                length_bin=item["length_bin"],
                word_count=wc,
                category=item["category"],
            )
        )
    return texts


def _parse_corpus_file(content: str) -> Tuple[Dict[str, str], str]:
    """Split a corpus file into metadata dict and body text."""
    meta: Dict[str, str] = {}
    body_lines: List[str] = []
    in_body = False

    for line in content.splitlines():
        stripped = line.strip()
        if not in_body and stripped.startswith("#"):
            if ":" in stripped:
                key, _, value = stripped.lstrip("#").strip().partition(":")
                meta[key.strip().lower()] = value.strip()
        elif stripped == "" and not in_body:
            continue
        else:
            in_body = True
            body_lines.append(line)

    return meta, "\n".join(body_lines)


def _infer_length_bin(word_count: int) -> str:
    if word_count <= 40:
        return "short"
    elif word_count <= 100:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def check_api_health(base_url: str) -> Tuple[bool, str]:
    """Check if the API server is reachable."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10.0)
        data = resp.json()
        if data.get("status") == "healthy":
            return True, "ok"
        return False, f"API unhealthy: {data}"
    except httpx.ConnectError:
        return False, f"Cannot connect to {base_url} — is the server running?"
    except Exception as exc:
        return False, str(exc)


def call_generate(
    client: httpx.Client,
    base_url: str,
    text: str,
    provider: str,
    model: str,
    max_iterations: int,
    target_violations: int,
) -> Dict[str, Any]:
    """Call the /generate endpoint and return the parsed response."""
    payload = {
        "text": text,
        "provider": provider,
        "model": model,
        "max_iterations": max_iterations,
        "target_violations": target_violations,
    }
    resp = client.post(f"{base_url}/generate", json=payload)
    resp.raise_for_status()
    return resp.json()


def call_analyse(
    client: httpx.Client,
    base_url: str,
    text: str,
) -> Dict[str, Any]:
    """Call the /analyse endpoint to get per-rule violation breakdown."""
    resp = client.post(f"{base_url}/analyse", json={"text": text})
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_single(
    client: httpx.Client,
    base_url: str,
    provider: str,
    model: str,
    eval_text: EvalText,
    max_iterations: int,
    target_violations: int,
) -> EvalResult:
    """Run a single (provider, model, text) evaluation via the API."""
    tier = MODEL_SIZE_TIERS.get(model, "unknown")

    t_start = time.monotonic()
    try:
        gen = call_generate(
            client,
            base_url,
            eval_text.text,
            provider,
            model,
            max_iterations,
            target_violations,
        )
        elapsed = time.monotonic() - t_start

        # Get per-rule breakdown by analysing the generated text
        violations_by_rule: Dict[str, int] = {}
        result_text = gen.get("result", "")
        if result_text:
            try:
                analysis = call_analyse(client, base_url, result_text)
                violations_by_rule = analysis.get("statistics", {}).get(
                    "violations_by_rule", {}
                )
            except Exception as exc:
                logger.warning("Analyse call failed for rule breakdown: %s", exc)

        return EvalResult(
            provider=gen.get("provider", provider),
            model=gen.get("model", model),
            model_size_tier=tier,
            text_id=eval_text.text_id,
            text_length_bin=eval_text.length_bin,
            text_word_count=eval_text.word_count,
            success=gen.get("success", False),
            final_violations=gen.get("final_violations", -1),
            total_iterations=gen.get("iterations", 0),
            final_escalation_level=gen.get("final_escalation_level", 0),
            elapsed_seconds=elapsed,
            violations_by_rule=violations_by_rule,
            faithfulness_score=gen.get("faithfulness_score"),
            hix=gen.get("hix"),
            stop_reason=gen.get("stop_reason"),
            error=None,
        )

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - t_start
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:200]
        logger.error(
            "HTTP %d for %s/%s/%s: %s",
            exc.response.status_code,
            provider,
            model,
            eval_text.text_id,
            detail,
        )
        return _error_result(
            provider,
            model,
            tier,
            eval_text,
            elapsed,
            f"HTTP {exc.response.status_code}: {detail}",
        )

    except Exception as exc:
        elapsed = time.monotonic() - t_start
        logger.error(
            "Run failed (%s/%s, %s): %s", provider, model, eval_text.text_id, exc
        )
        return _error_result(provider, model, tier, eval_text, elapsed, str(exc))


def _error_result(
    provider: str,
    model: str,
    tier: str,
    eval_text: EvalText,
    elapsed: float,
    error: str,
) -> EvalResult:
    return EvalResult(
        provider=provider,
        model=model,
        model_size_tier=tier,
        text_id=eval_text.text_id,
        text_length_bin=eval_text.length_bin,
        text_word_count=eval_text.word_count,
        success=False,
        final_violations=-1,
        total_iterations=0,
        final_escalation_level=0,
        elapsed_seconds=elapsed,
        violations_by_rule={},
        faithfulness_score=None,
        hix=None,
        stop_reason=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, result: EvalResult) -> None:
    """Append one result as a JSON line (crash-safe)."""
    line = json.dumps(result.to_dict(), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_completed_runs(jsonl_path: Path) -> Set[Tuple[str, str, str]]:
    """Load set of (provider, model, text_id) from existing JSONL."""
    completed: Set[Tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add((record["provider"], record["model"], record["text_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------


def generate_summary_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Read JSONL results and write a grouped CSV summary."""
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        logger.warning("No records found in %s — skipping summary", jsonl_path)
        return

    # Group by (provider, model, size_tier, length_bin)
    groups: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for rec in records:
        key = (
            rec["provider"],
            rec["model"],
            rec["model_size_tier"],
            rec["text_length_bin"],
        )
        groups[key].append(rec)

    rows = []
    for (provider, model, tier, length_bin), group in sorted(groups.items()):
        n = len(group)
        successes = sum(1 for r in group if r.get("success"))
        valid = [r for r in group if r.get("final_violations", -1) >= 0]
        avg_violations = sum(r["final_violations"] for r in valid) / max(1, len(valid))
        avg_iterations = sum(r.get("total_iterations", 0) for r in group) / max(1, n)
        avg_seconds = sum(r.get("elapsed_seconds", 0) for r in group) / max(1, n)

        faith_scores = [
            r["faithfulness_score"]
            for r in group
            if r.get("faithfulness_score") is not None
        ]
        avg_faithfulness = (
            round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else None
        )

        hix_scores = [r["hix"] for r in group if r.get("hix") is not None]
        avg_hix = round(sum(hix_scores) / len(hix_scores), 1) if hix_scores else None

        rows.append(
            {
                "provider": provider,
                "model": model,
                "size_tier": tier,
                "length_bin": length_bin,
                "n": n,
                "success_rate": round(successes / n, 2),
                "avg_violations": round(avg_violations, 1),
                "avg_iterations": round(avg_iterations, 1),
                "avg_seconds": round(avg_seconds, 1),
                "avg_faithfulness": avg_faithfulness or "",
                "avg_hix": avg_hix or "",
            }
        )

    fieldnames = [
        "provider",
        "model",
        "size_tier",
        "length_bin",
        "n",
        "success_rate",
        "avg_violations",
        "avg_iterations",
        "avg_seconds",
        "avg_faithfulness",
        "avg_hix",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary written to %s (%d rows)", csv_path, len(rows))


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_sigint(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        sys.exit(1)
    _shutdown_requested = True
    logger.info("Shutdown requested -- finishing current run and writing summary...")


# ---------------------------------------------------------------------------
# CLI + main loop
# ---------------------------------------------------------------------------


def build_run_matrix(
    corpus: List[EvalText],
    providers: Optional[List[str]],
    models: Optional[List[str]],
    lengths: Optional[List[str]],
) -> List[Tuple[str, str, EvalText]]:
    """Build list of (provider, model, EvalText) tuples to evaluate.

    If --models is given, provider is inferred from MODEL_TO_PROVIDER.
    If --providers is given (without --models), all models for those providers are used.
    If neither is given, defaults to all Mistral models (backward compat).
    """
    runs: List[Tuple[str, str, EvalText]] = []

    if models:
        # Explicit model list — infer provider per model
        pairs: List[Tuple[str, str]] = []
        for m in models:
            provider = MODEL_TO_PROVIDER.get(m)
            if not provider:
                logger.warning("Unknown model %s — skipping", m)
                continue
            pairs.append((provider, m))
    elif providers:
        # Explicit provider list — use all models for each
        pairs = []
        for p in providers:
            for m in MODELS_BY_PROVIDER.get(p, []):
                pairs.append((p, m))
    else:
        # Default: Mistral models (backward compat)
        pairs = [("mistral", m) for m in MODELS_BY_PROVIDER["mistral"]]

    for provider, model in pairs:
        for text in corpus:
            if lengths and text.length_bin not in lengths:
                continue
            runs.append((provider, model, text))

    return runs


def _all_model_names() -> List[str]:
    """Flat list of all known model names for help text."""
    return [m for models in MODELS_BY_PROVIDER.values() for m in models]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models across providers for Leichte Sprache generation",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Base URL of the running API (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Providers to evaluate (default: mistral). "
        f"Choices: {', '.join(ALL_PROVIDERS)}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to evaluate (provider is inferred). "
        "Overrides --providers. "
        f"Choices: {', '.join(_all_model_names())}",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        help="Filter text lengths (short, medium, long)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max iterations per generation (default: 10)",
    )
    parser.add_argument(
        "--target-violations",
        type=int,
        default=2,
        help="Target violation count (default: 2)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing JSONL file (skip completed runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned runs without executing",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=10,
        help="Seconds to wait between runs to avoid rate limits (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Load corpus
    corpus_dir = Path(__file__).parent / "eval_corpus"
    corpus = load_corpus(corpus_dir)
    logger.info("Loaded %d texts from corpus", len(corpus))

    # Build run matrix
    runs = build_run_matrix(corpus, args.providers, args.models, args.lengths)

    if not runs:
        logger.error("No runs to execute. Check --providers/--models/--lengths filters.")
        logger.info("Available providers: %s", ", ".join(ALL_PROVIDERS))
        for p, models in MODELS_BY_PROVIDER.items():
            logger.info("  %s: %s", p, ", ".join(models))
        sys.exit(1)

    # Handle resume
    completed: Set[Tuple[str, str, str]] = set()
    if args.resume:
        resume_path = Path(args.resume)
        completed = load_completed_runs(resume_path)
        logger.info("Resuming: %d runs already completed", len(completed))

    # Filter out completed runs
    pending_runs = [(p, m, t) for p, m, t in runs if (p, m, t.text_id) not in completed]

    # Dry run
    if args.dry_run:
        logger.info(
            "Dry run -- %s runs planned (%s already done):", len(pending_runs), len(completed)
        )
        logger.info(
            "%-12s %-28s %-8s %-28s %-8s %-6s",
            "Provider", "Model", "Tier", "Text ID", "Length", "Words",
        )
        logger.info("-" * 90)
        for provider, model, text in pending_runs:
            tier = MODEL_SIZE_TIERS.get(model, "?")
            logger.info(
                "%-12s %-28s %-8s %-28s %-8s %-6s",
                provider, model, tier, text.text_id, text.length_bin, text.word_count,
            )
        logger.info("Total: %s runs", len(pending_runs))
        return

    # Check API server is reachable
    available, reason = check_api_health(args.api_url)
    if not available:
        logger.error("API not reachable: %s", reason)
        logger.error("Start the server first: python api_main.py")
        sys.exit(1)

    if not pending_runs:
        logger.info("All runs already completed. Nothing to do.")
        return

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        jsonl_path = Path(args.resume)
    else:
        jsonl_path = output_dir / f"eval_{timestamp}.jsonl"
    csv_path = jsonl_path.with_name(jsonl_path.stem + "_summary.csv")

    # Install signal handler
    signal.signal(signal.SIGINT, _handle_sigint)

    # Summarize providers in this run
    providers_in_run = sorted({p for p, _, _ in pending_runs})
    logger.info("Starting evaluation: %s runs", len(pending_runs))
    logger.info("  API: %s", args.api_url)
    logger.info("  Providers: %s", ", ".join(providers_in_run))
    logger.info("  Output: %s", jsonl_path)
    logger.info("  Max iterations: %s", args.max_iterations)
    logger.info("  Target violations: %s", args.target_violations)
    logger.info("  Delay between runs: %ss", args.delay)

    client = httpx.Client(timeout=HTTP_TIMEOUT)

    completed_count = len(completed)
    total_count = len(runs)

    is_first_run = True

    try:
        for provider, model, eval_text in pending_runs:
            if _shutdown_requested:
                break

            # Rate-limit delay between runs (skip before first)
            if not is_first_run and args.delay > 0:
                time.sleep(args.delay)
            is_first_run = False

            completed_count += 1
            tier = MODEL_SIZE_TIERS.get(model, "?")
            logger.info(
                "[%s/%s] %s/%s (%s) x %s (%s, %sw) ...",
                completed_count, total_count, provider, model, tier,
                eval_text.text_id, eval_text.length_bin, eval_text.word_count,
            )

            result = run_single(
                client,
                args.api_url,
                provider,
                model,
                eval_text,
                args.max_iterations,
                args.target_violations,
            )
            append_jsonl(jsonl_path, result)

            if result.error:
                logger.error("ERROR: %s", result.error[:80])
            else:
                status = "OK" if result.success else "FAIL"
                logger.info(
                    "%s v=%s i=%s t=%.1fs",
                    status, result.final_violations,
                    result.total_iterations, result.elapsed_seconds,
                )
    finally:
        client.close()

    # Generate summary
    if jsonl_path.exists():
        generate_summary_csv(jsonl_path, csv_path)
        logger.info("Results: %s", jsonl_path)
        logger.info("Summary: %s", csv_path)
    else:
        logger.info("No results written.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
