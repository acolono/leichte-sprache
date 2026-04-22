"""Training adapter for the perplexity_saetze NLTK Kneser-Ney n-gram model.

Follows the CLI contract from tools/ml so that:
    python -m tools.ml train perplexity_saetze
discovers and runs training through the unified interface.
"""

from pathlib import Path

MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "leichte_sprache_korpus.txt"

# METADATA for CLI discovery (tools/ml --list)
METADATA = {
    "model_type": "nltk-ngram",
    "framework": "nltk",
    "description": "NLTK Kneser-Ney n-gram perplexity model for sentence complexity",
}


def train(output_dir: Path, device: str, seed: int) -> dict:
    """Train the Kneser-Ney n-gram perplexity model.

    Delegates to the inner functions in train.py, bypassing main() to avoid
    argparse conflicts with the Typer CLI.

    Args:
        output_dir: Staging directory for training output.
        device: Device string -- accepted for contract compliance but unused
                (NLTK n-gram training is CPU-only and deterministic).
        seed: Random seed -- accepted for contract compliance but unused
              (Kneser-Ney interpolation is deterministic).

    Returns:
        Dict with training metrics.

    Raises:
        FileNotFoundError: If the training corpus is missing.
    """
    # Heavy imports inside function body to keep --list startup fast.
    from regeln.perplexity_saetze.train import (
        ensure_nltk_data,
        load_corpus,
        save_model,
        tokenize_sentences,
        train_model,
    )

    corpus_path = DEFAULT_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Provide training corpus via --data flag."
        )

    ensure_nltk_data()
    sentences = load_corpus(str(corpus_path))
    tokenized = tokenize_sentences(sentences)
    model = train_model(tokenized, n_gram_size=3)
    save_model(model, str(output_dir / "perplexity_model.pkl"))

    return {
        "status": "completed",
        "vocab_size": len(model.vocab),
        "sentences_trained": len(tokenized),
    }


def evaluate(model_dir: Path, **kwargs) -> dict:
    """Evaluate the Kneser-Ney perplexity model on a corpus sample.

    Since the n-gram perplexity model is unsupervised, there are no
    precision/recall/F1 metrics. Instead we report average perplexity
    across the evaluation corpus, vocabulary size, and sentence count.

    Args:
        model_dir: Path to directory containing perplexity_model.pkl.
        **kwargs:  Accepts optional ``data_path`` for a custom corpus.

    Returns:
        Dict with avg_perplexity, vocab_size, sentences_evaluated.
    """
    try:
        import pickle

        import nltk

        from regeln.perplexity_saetze.train import (
            ensure_nltk_data,
            load_corpus,
            tokenize_sentences,
        )

        model_dir = Path(model_dir)
        model_path = model_dir / "perplexity_model.pkl"

        if not model_path.exists():
            return {"status": "error", "message": f"Model not found: {model_path}"}

        # Load model
        with open(model_path, "rb") as fh:
            data = pickle.load(fh)

        model = data["model"] if isinstance(data, dict) else data

        # Load evaluation corpus
        data_path = kwargs.get("data_path")
        if data_path is not None:
            corpus_path = Path(data_path)
        else:
            corpus_path = DEFAULT_CORPUS

        if not corpus_path.exists():
            return {"status": "error", "message": f"Corpus not found: {corpus_path}"}

        ensure_nltk_data()
        sentences = load_corpus(str(corpus_path))
        tokenized = tokenize_sentences(sentences)

        # Compute perplexity per sentence
        perplexities = []
        errors = 0
        for tokens in tokenized:
            try:
                ppl = model.perplexity(tokens)
                if ppl != float("inf") and ppl == ppl:  # skip inf and nan
                    perplexities.append(ppl)
            except Exception:
                errors += 1

        if not perplexities:
            return {"status": "error", "message": "No valid perplexity scores computed"}

        avg_ppl = sum(perplexities) / len(perplexities)

        return {
            "status": "completed",
            "avg_perplexity": round(avg_ppl, 2),
            "vocab_size": len(model.vocab),
            "sentences_evaluated": len(perplexities),
            "sentences_skipped": errors + (len(tokenized) - len(perplexities) - errors),
            "classification_report": "N/A (unsupervised n-gram model -- no P/R/F1)",
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
