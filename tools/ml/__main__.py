"""CLI entry point for ML rule training and evaluation.

Usage:
    python -m tools.ml --help
    python -m tools.ml train --list
    python -m tools.ml train <rule> [--device auto] [--seed 42] [--promote]
    python -m tools.ml evaluate <rule> [--model-dir path] [--json] [--verbose]
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tools.ml.device import detect_device, set_seed
from tools.ml.discovery import discover_trainable_rules, get_training_status, load_metadata
from tools.ml.staging import get_staging_dir, promote_model

app = typer.Typer(
    name="ml",
    help="ML rule training and evaluation toolkit",
    rich_markup_mode="rich",
)
console = Console()


def _show_rule_list() -> None:
    """Display a Rich table of all discoverable trainable rules."""
    rules = discover_trainable_rules()

    if not rules:
        console.print("[yellow]No trainable rules found.[/yellow]")
        console.print("Rules need a train_config.py in their directory.")
        return

    table = Table(title="Trainable ML Rules")
    table.add_column("Rule", style="cyan")
    table.add_column("Model Type", style="green")
    table.add_column("Framework", style="yellow")
    table.add_column("Status", style="bold")

    for name, rule_dir in rules.items():
        metadata = load_metadata(name)
        status = get_training_status(rule_dir)
        status_style = "green" if status == "trained" else "red"
        table.add_row(
            name,
            metadata.get("model_type", "unknown"),
            metadata.get("framework", "unknown"),
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)


@app.command()
def train(
    rule: str = typer.Argument(None, help="Rule name to train"),
    list_rules: bool = typer.Option(False, "--list", help="List all trainable rules"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, mps"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    promote: bool = typer.Option(False, help="Promote staged model after training"),
) -> None:
    """Train an ML rule model."""
    if list_rules:
        _show_rule_list()
        return

    if rule is None:
        console.print("[red]Error:[/red] Specify a rule name or use --list")
        raise typer.Exit(1)

    # Validate rule exists
    rules = discover_trainable_rules()
    if rule not in rules:
        console.print(f"[red]Error:[/red] Rule '{rule}' not found or has no train_config.py")
        available = ", ".join(rules.keys()) if rules else "none"
        console.print(f"Available rules: {available}")
        raise typer.Exit(1)

    rule_dir = rules[rule]

    # Set up device and seed
    try:
        resolved_device = detect_device(device)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    set_seed(seed)
    console.print(f"[green]Device:[/green] {resolved_device}")
    console.print(f"[green]Seed:[/green] {seed}")

    # Get staging directory for output
    staging_dir = get_staging_dir(rule_dir)
    console.print(f"[green]Output:[/green] {staging_dir}")

    # Import and call the rule's train function
    try:
        train_config = importlib.import_module(f"regeln.{rule}.train_config")
        train_fn = getattr(train_config, "train", None)
        if train_fn is None:
            console.print(
                f"[red]Error:[/red] regeln.{rule}.train_config has no train() function"
            )
            raise typer.Exit(1)

        console.print(f"\n[bold]Training {rule}...[/bold]\n")
        metrics = train_fn(output_dir=staging_dir, device=resolved_device, seed=seed)

        # Display metrics
        if metrics:
            metrics_table = Table(title="Training Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            for key, value in metrics.items():
                metrics_table.add_row(
                    key,
                    f"{value:.4f}" if isinstance(value, float) else str(value),
                )
            console.print(metrics_table)

        console.print(f"\n[green]Training complete.[/green] Output in {staging_dir}")

    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        raise typer.Exit(1)

    # Promote if requested
    if promote:
        try:
            backup_dir = promote_model(rule_dir)
            console.print(f"[green]Model promoted.[/green] Backup at {backup_dir}")
            console.print("[yellow]Note:[/yellow] Restart the API server to use the new model.")
        except FileNotFoundError as e:
            console.print(f"[red]Promote failed:[/red] {e}")
            raise typer.Exit(1)


def _resolve_model_dir(rule_dir: Path, model_dir: str | None) -> Path:
    """Resolve model directory: explicit path, or deployed model with smart fallback."""
    if model_dir is not None:
        return Path(model_dir)
    candidate = rule_dir / "model"
    # Only use model/ subdir if it exists AND has real files (not just .gitkeep)
    if candidate.is_dir() and any(
        f.name != ".gitkeep" for f in candidate.iterdir() if f.is_file()
    ):
        return candidate
    return rule_dir


def _run_test_suite(rule: str) -> dict[str, int] | None:
    """Run the test suite for a rule via subprocess and parse results.

    Returns dict with total/passed/failed counts, or None on failure.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    test_runner = project_root / "test-suite" / "test_runner.py"
    if not test_runner.exists():
        return None

    try:
        result = subprocess.run(
            [sys.executable, str(test_runner), rule],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_root),
        )
        # Parse SUMMARY line: "Total: N tests, N passed, N failed"
        for line in result.stderr.splitlines() + result.stdout.splitlines():
            match = re.search(
                r"Total:\s*(\d+)\s*tests?,\s*(\d+)\s*passed,\s*(\d+)\s*failed", line
            )
            if match:
                return {
                    "total": int(match.group(1)),
                    "passed": int(match.group(2)),
                    "failed": int(match.group(3)),
                }
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _evaluate_model(rule: str, eval_dir: Path, data: str | None) -> dict | None:
    """Import and run a rule's evaluate function. Returns metrics dict or None."""
    train_config = importlib.import_module(f"regeln.{rule}.train_config")
    evaluate_fn = getattr(train_config, "evaluate", None)
    if evaluate_fn is None:
        return None

    eval_kwargs: dict = {}
    if data is not None:
        eval_kwargs["data_path"] = Path(data)

    metrics = evaluate_fn(model_dir=eval_dir, **eval_kwargs)
    return metrics


def _display_metrics_table(rule: str, metrics: dict, label: str = "") -> str | None:
    """Display Rich metrics table. Returns classification_report if present."""
    report = metrics.pop("classification_report", None)

    title = f"Evaluation Metrics: {rule}"
    if label:
        title += f" ({label})"
    metrics_table = Table(title=title)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    for key, value in metrics.items():
        if key == "status":
            continue
        if isinstance(value, float):
            metrics_table.add_row(key, f"{value:.4f}")
        else:
            metrics_table.add_row(key, str(value))

    console.print(metrics_table)
    return report


def _display_comparison_table(deployed: dict, staged: dict) -> None:
    """Display side-by-side comparison of deployed vs staged metrics."""
    # Metrics where higher is better
    higher_is_better = {"precision", "recall", "f1", "accuracy", "avg_perplexity_inv"}
    # Metrics where lower is better
    lower_is_better = {"loss", "avg_perplexity"}

    table = Table(title="Comparison: Deployed vs Staged")
    table.add_column("Metric", style="cyan")
    table.add_column("Deployed", style="white")
    table.add_column("Staged", style="white")
    table.add_column("Delta", style="white")

    all_keys = list(dict.fromkeys(list(deployed.keys()) + list(staged.keys())))
    for key in all_keys:
        if key in ("status", "classification_report"):
            continue
        d_val = deployed.get(key)
        s_val = staged.get(key)

        d_str = f"{d_val:.4f}" if isinstance(d_val, float) else str(d_val) if d_val is not None else "-"
        s_str = f"{s_val:.4f}" if isinstance(s_val, float) else str(s_val) if s_val is not None else "-"

        if isinstance(d_val, (int, float)) and isinstance(s_val, (int, float)):
            delta = s_val - d_val
            # Determine color
            if key in higher_is_better:
                color = "green" if delta > 0 else "red" if delta < 0 else "yellow"
            elif key in lower_is_better:
                color = "green" if delta < 0 else "red" if delta > 0 else "yellow"
            else:
                color = "yellow"
            sign = "+" if delta > 0 else ""
            delta_str = f"[{color}]{sign}{delta:.4f}[/{color}]" if isinstance(delta, float) else f"[{color}]{sign}{delta}[/{color}]"
        else:
            delta_str = "-"

        table.add_row(key, d_str, s_str, delta_str)

    console.print(table)


def _display_test_suite_table(rule: str, test_results: dict) -> None:
    """Display test suite results as a Rich table."""
    table = Table(title=f"Test Suite Results: {rule}")
    table.add_column("Total", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Result", style="bold")

    result_str = (
        "[green]PASSED[/green]"
        if test_results["failed"] == 0 and test_results["total"] > 0
        else "[red]FAILED[/red]"
        if test_results["failed"] > 0
        else "[yellow]NO TESTS[/yellow]"
    )

    table.add_row(
        str(test_results["total"]),
        str(test_results["passed"]),
        str(test_results["failed"]),
        result_str,
    )
    console.print(table)


@app.command()
def evaluate(
    rule: str = typer.Argument(..., help="Rule name to evaluate"),
    model_dir: str = typer.Option(None, help="Model directory (default: deployed model)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-class metric breakdown"),
    data: str = typer.Option(None, "--data", help="Path to custom evaluation dataset"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Evaluate a trained ML rule model."""
    # Validate rule exists
    rules = discover_trainable_rules()
    if rule not in rules:
        console.print(f"[red]Error:[/red] Rule '{rule}' not found or has no train_config.py")
        raise typer.Exit(1)

    rule_dir = rules[rule]
    eval_dir = _resolve_model_dir(rule_dir, model_dir)

    if not eval_dir.exists():
        console.print(f"[red]Error:[/red] Model directory not found: {eval_dir}")
        raise typer.Exit(1)

    # --- Evaluate deployed (or explicit) model ---
    try:
        train_config = importlib.import_module(f"regeln.{rule}.train_config")
        evaluate_fn = getattr(train_config, "evaluate", None)
        if evaluate_fn is None:
            console.print(
                f"[red]Error:[/red] regeln.{rule}.train_config has no evaluate() function"
            )
            raise typer.Exit(1)

        if not json_output:
            console.print(f"[bold]Evaluating {rule}...[/bold]\n")

        metrics = _evaluate_model(rule, eval_dir, data)

    except Exception as e:
        console.print(f"[red]Evaluation failed for {rule}:[/red] {e}")
        raise typer.Exit(1)

    if not metrics:
        console.print("[yellow]No metrics returned.[/yellow]")
        return

    # Handle error status gracefully
    if metrics.get("status") == "error":
        if json_output:
            print(json.dumps({"rule": rule, "error": metrics.get("message", "Unknown error")}, indent=2))
        else:
            console.print(f"[yellow]Warning:[/yellow] {metrics.get('message', 'Unknown error')}")
        return

    # --- Staged vs deployed comparison (only when model_dir not explicit) ---
    staged_metrics: dict | None = None
    if model_dir is None:
        staging_dir = get_staging_dir(rule_dir)
        # Check for actual staged files (not just the empty dir get_staging_dir creates)
        if staging_dir.exists() and any(staging_dir.iterdir()):
            try:
                staged_metrics = _evaluate_model(rule, staging_dir, data)
                if staged_metrics and staged_metrics.get("status") == "error":
                    staged_metrics = None
            except Exception:
                staged_metrics = None

    # --- Run test suite ---
    test_results = _run_test_suite(rule)

    # --- JSON output mode ---
    if json_output:
        # Remove classification_report from metrics for clean JSON
        deployed_for_json = {k: v for k, v in metrics.items() if k != "classification_report"}
        staged_for_json = None
        if staged_metrics:
            staged_for_json = {k: v for k, v in staged_metrics.items() if k != "classification_report"}

        output = {
            "rule": rule,
            "model_metrics": deployed_for_json,
            "staged_metrics": staged_for_json,
            "test_suite": test_results,
        }
        print(json.dumps(output, indent=2))
        return

    # --- Rich output mode ---
    # 1. Model metrics table
    report = _display_metrics_table(rule, metrics)

    # 2. Verbose classification report
    if verbose and report and report != "N/A (unsupervised n-gram model -- no P/R/F1)":
        from rich.panel import Panel

        console.print()
        console.print(Panel(report.strip(), title="Per-Class Breakdown", border_style="dim"))
    elif verbose and report:
        console.print(f"\n[dim]{report}[/dim]")

    # 3. Comparison table (if staged model exists)
    if staged_metrics:
        console.print()
        # Keep a copy without classification_report for comparison
        deployed_copy = {k: v for k, v in metrics.items() if k != "classification_report"}
        staged_copy = {k: v for k, v in staged_metrics.items() if k != "classification_report"}
        _display_comparison_table(deployed_copy, staged_copy)

    # 4. Test suite results table
    if test_results is not None:
        console.print()
        _display_test_suite_table(rule, test_results)

    console.print("\n[green]Evaluation complete.[/green]")


if __name__ == "__main__":
    app()
