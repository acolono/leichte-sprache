"""
Hohenheimer Verständlichkeitsindex (HIX) Computation Module

Computes the HIX score (0–20 scale) for German texts based on four readability
formulas and five text parameters. Each sub-score is normalized to 0–10 via
linear interpolation against Hohenheim corpus benchmarks, then averaged:

    HIX = formula_score + parameter_score

where formula_score = mean of four normalized formula values (0–10) and
parameter_score = mean of five normalized parameter values (0–10).
"""

import math
from enum import Enum
from typing import Any, Dict, List, Tuple

import pyphen
from pydantic import BaseModel
from spacy.tokens import Doc

# ---------------------------------------------------------------------------
# Syllable counter (pyphen-based, same approach as regeln/kurze_woerter)
# ---------------------------------------------------------------------------

_hyphenator = pyphen.Pyphen(lang="de_DE")


def _count_syllables(word: str) -> int:
    """Count syllables, splitting hyphenated compounds into parts first."""
    if "-" in word:
        total = 0
        for part in word.split("-"):
            if part:
                total += len(_hyphenator.inserted(part).split("-"))
        return total
    return len(_hyphenator.inserted(word).split("-"))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class FormulaResult(BaseModel):
    raw: float
    normalized: float


class ParameterResult(BaseModel):
    raw: float
    normalized: float


class FormulaScores(BaseModel):
    amstad: FormulaResult
    wiener_sachtextformel: FormulaResult
    smog_de: FormulaResult
    lix: FormulaResult


class TextParameters(BaseModel):
    avg_sentence_length: ParameterResult
    avg_word_length_syllables: ParameterResult
    pct_long_words: ParameterResult
    pct_polysyllabic: ParameterResult
    pct_monosyllabic: ParameterResult


class HixRating(str, Enum):
    EXPERT = "expert"
    COMPLEX = "complex"
    READABLE = "readable"
    PLAIN = "plain"
    SIMPLE = "simple"
    EASY = "easy"


class HixResult(BaseModel):
    hix: float
    formula_score: float
    parameter_score: float
    formulas: FormulaScores
    parameters: TextParameters
    rating: HixRating


# ---------------------------------------------------------------------------
# Normalization benchmark tables  (raw_value, normalized_score)
# Based on Hohenheim corpus data.
# ---------------------------------------------------------------------------

_BENCHMARKS_AMSTAD: List[Tuple[float, float]] = [
    (0, 0),
    (20, 2),
    (40, 4),
    (60, 6),
    (70, 7),
    (80, 8),
    (90, 9),
    (100, 10),
]

_BENCHMARKS_WIENER: List[Tuple[float, float]] = [
    (15, 0),
    (12, 2),
    (10, 4),
    (8, 6),
    (6, 7),
    (4, 8),
    (2, 9),
    (0, 10),
]

_BENCHMARKS_SMOG: List[Tuple[float, float]] = [
    (18, 0),
    (15, 2),
    (12, 4),
    (9, 6),
    (7, 7),
    (5, 8),
    (3, 9),
    (1, 10),
]

_BENCHMARKS_LIX: List[Tuple[float, float]] = [
    (70, 0),
    (60, 2),
    (50, 4),
    (40, 6),
    (35, 7),
    (30, 8),
    (25, 9),
    (20, 10),
]

_BENCHMARKS_AVG_SENT_LEN: List[Tuple[float, float]] = [
    (25, 0),
    (20, 3),
    (15, 5),
    (10, 7),
    (8, 8),
    (6, 9),
    (4, 10),
]

_BENCHMARKS_AVG_WORD_LEN_SYLL: List[Tuple[float, float]] = [
    (3.0, 0),
    (2.5, 3),
    (2.0, 5),
    (1.8, 7),
    (1.5, 8),
    (1.3, 9),
    (1.0, 10),
]

_BENCHMARKS_PCT_LONG_WORDS: List[Tuple[float, float]] = [
    (50, 0),
    (40, 3),
    (30, 5),
    (20, 7),
    (15, 8),
    (10, 9),
    (5, 10),
]

_BENCHMARKS_PCT_POLYSYLLABIC: List[Tuple[float, float]] = [
    (40, 0),
    (30, 3),
    (20, 5),
    (15, 7),
    (10, 8),
    (5, 9),
    (0, 10),
]

# Monosyllabic: higher raw = simpler (inverted direction)
_BENCHMARKS_PCT_MONOSYLLABIC: List[Tuple[float, float]] = [
    (10, 0),
    (20, 3),
    (30, 5),
    (40, 7),
    (50, 8),
    (60, 9),
    (80, 10),
]


def _normalize(raw: float, benchmarks: List[Tuple[float, float]]) -> float:
    """Linear interpolation between benchmark stützstellen.

    Benchmarks are sorted by raw value internally.  Values beyond the
    table edges are clamped to the nearest endpoint's normalized score.
    """
    # Sort by raw value ascending for interpolation
    pts = sorted(benchmarks, key=lambda p: p[0])

    if raw <= pts[0][0]:
        return pts[0][1]
    if raw >= pts[-1][0]:
        return pts[-1][1]

    for i in range(len(pts) - 1):
        r0, n0 = pts[i]
        r1, n1 = pts[i + 1]
        if r0 <= raw <= r1:
            t = (raw - r0) / (r1 - r0) if r1 != r0 else 0.0
            return n0 + t * (n1 - n0)

    return pts[-1][1]  # fallback


# ---------------------------------------------------------------------------
# Text statistics extraction from spaCy Doc
# ---------------------------------------------------------------------------


def _extract_text_stats(doc: Doc) -> Dict[str, Any]:
    """Extract word/sentence/syllable statistics from a spaCy Doc.

    Token filter: exclude punctuation, spaces, and pure digit tokens
    (same logic as regeln/kurze_woerter/regel.py:60-63).
    """
    words: List[str] = []
    syllable_counts: List[int] = []
    char_counts: List[int] = []

    for token in doc:
        if token.is_punct or token.is_space or token.is_digit:
            continue
        text = token.text
        words.append(text)
        syllable_counts.append(_count_syllables(text))
        char_counts.append(len(text))

    sentence_count = max(1, sum(1 for _ in doc.sents))
    word_count = len(words)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_counts": syllable_counts,
        "char_counts": char_counts,
    }


# ---------------------------------------------------------------------------
# Four readability formulas
# ---------------------------------------------------------------------------


def _compute_amstad(stats: Dict[str, Any]) -> FormulaResult:
    """Amstad formula (German Reading Ease): 180 - ASL - 58.5 * ASW."""
    wc = stats["word_count"]
    sc = stats["sentence_count"]
    if wc == 0:
        return FormulaResult(raw=0.0, normalized=0.0)

    asl = wc / sc  # average sentence length
    asw = sum(stats["syllable_counts"]) / wc  # avg syllables per word
    raw = 180.0 - asl - 58.5 * asw
    return FormulaResult(
        raw=round(raw, 2), normalized=round(_normalize(raw, _BENCHMARKS_AMSTAD), 2)
    )


def _compute_wiener(stats: Dict[str, Any]) -> FormulaResult:
    """Wiener Sachtextformel (1st variant).

    nWS1 = 0.1935 * MS + 0.1672 * SL + 0.1297 * IW - 0.0327 * ES - 0.875
    MS = % words >= 3 syllables
    SL = avg sentence length (words)
    IW = % words > 6 characters
    ES = % monosyllabic words
    """
    wc = stats["word_count"]
    sc = stats["sentence_count"]
    if wc == 0:
        return FormulaResult(raw=0.0, normalized=0.0)

    ms = 100.0 * sum(1 for s in stats["syllable_counts"] if s >= 3) / wc
    sl = wc / sc
    iw = 100.0 * sum(1 for c in stats["char_counts"] if c > 6) / wc
    es = 100.0 * sum(1 for s in stats["syllable_counts"] if s == 1) / wc

    raw = 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875
    return FormulaResult(
        raw=round(raw, 2), normalized=round(_normalize(raw, _BENCHMARKS_WIENER), 2)
    )


def _compute_smog(stats: Dict[str, Any]) -> FormulaResult:
    """SMOG (German adaptation): 3 + sqrt(MS3).

    MS3 = polysyllabic words (>=3 syllables), proportionally scaled to
    30 sentences.
    """
    wc = stats["word_count"]
    sc = stats["sentence_count"]
    if wc == 0:
        return FormulaResult(raw=0.0, normalized=0.0)

    polysyllabic = sum(1 for s in stats["syllable_counts"] if s >= 3)
    # Scale to 30-sentence base
    ms3 = polysyllabic * (30.0 / sc) if sc > 0 else 0.0
    raw = 3.0 + math.sqrt(ms3)
    return FormulaResult(
        raw=round(raw, 2), normalized=round(_normalize(raw, _BENCHMARKS_SMOG), 2)
    )


def _compute_lix(stats: Dict[str, Any]) -> FormulaResult:
    """LIX = LW + ASL.

    LW = % words > 6 characters
    ASL = average sentence length (words)
    """
    wc = stats["word_count"]
    sc = stats["sentence_count"]
    if wc == 0:
        return FormulaResult(raw=0.0, normalized=0.0)

    lw = 100.0 * sum(1 for c in stats["char_counts"] if c > 6) / wc
    asl = wc / sc
    raw = lw + asl
    return FormulaResult(
        raw=round(raw, 2), normalized=round(_normalize(raw, _BENCHMARKS_LIX), 2)
    )


# ---------------------------------------------------------------------------
# Five text parameters
# ---------------------------------------------------------------------------


def _compute_parameters(stats: Dict[str, Any]) -> TextParameters:
    wc = stats["word_count"]
    sc = stats["sentence_count"]
    if wc == 0:
        zero = ParameterResult(raw=0.0, normalized=10.0)
        return TextParameters(
            avg_sentence_length=zero,
            avg_word_length_syllables=zero,
            pct_long_words=zero,
            pct_polysyllabic=zero,
            pct_monosyllabic=zero,
        )

    asl = wc / sc
    asw = sum(stats["syllable_counts"]) / wc
    pct_long = 100.0 * sum(1 for c in stats["char_counts"] if c > 6) / wc
    pct_poly = 100.0 * sum(1 for s in stats["syllable_counts"] if s >= 3) / wc
    pct_mono = 100.0 * sum(1 for s in stats["syllable_counts"] if s == 1) / wc

    return TextParameters(
        avg_sentence_length=ParameterResult(
            raw=round(asl, 2),
            normalized=round(_normalize(asl, _BENCHMARKS_AVG_SENT_LEN), 2),
        ),
        avg_word_length_syllables=ParameterResult(
            raw=round(asw, 2),
            normalized=round(_normalize(asw, _BENCHMARKS_AVG_WORD_LEN_SYLL), 2),
        ),
        pct_long_words=ParameterResult(
            raw=round(pct_long, 2),
            normalized=round(_normalize(pct_long, _BENCHMARKS_PCT_LONG_WORDS), 2),
        ),
        pct_polysyllabic=ParameterResult(
            raw=round(pct_poly, 2),
            normalized=round(_normalize(pct_poly, _BENCHMARKS_PCT_POLYSYLLABIC), 2),
        ),
        pct_monosyllabic=ParameterResult(
            raw=round(pct_mono, 2),
            normalized=round(_normalize(pct_mono, _BENCHMARKS_PCT_MONOSYLLABIC), 2),
        ),
    )


# ---------------------------------------------------------------------------
# HIX rating
# ---------------------------------------------------------------------------


def _rate(hix: float) -> HixRating:
    if hix < 5:
        return HixRating.EXPERT
    if hix < 10:
        return HixRating.COMPLEX
    if hix < 15:
        return HixRating.READABLE
    if hix < 16:
        return HixRating.PLAIN
    if hix < 18:
        return HixRating.SIMPLE
    return HixRating.EASY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_hix(doc: Doc) -> HixResult:
    """Compute the HIX score for a spaCy Doc.

    Returns a HixResult with the overall score (0–20), sub-scores, and rating.
    """
    stats = _extract_text_stats(doc)

    amstad = _compute_amstad(stats)
    wiener = _compute_wiener(stats)
    smog = _compute_smog(stats)
    lix = _compute_lix(stats)

    formula_score = round(
        (amstad.normalized + wiener.normalized + smog.normalized + lix.normalized) / 4,
        2,
    )

    params = _compute_parameters(stats)
    param_values = [
        params.avg_sentence_length.normalized,
        params.avg_word_length_syllables.normalized,
        params.pct_long_words.normalized,
        params.pct_polysyllabic.normalized,
        params.pct_monosyllabic.normalized,
    ]
    parameter_score = round(sum(param_values) / len(param_values), 2)

    hix = round(formula_score + parameter_score, 2)
    # Clamp to 0–20
    hix = max(0.0, min(20.0, hix))

    return HixResult(
        hix=hix,
        formula_score=formula_score,
        parameter_score=parameter_score,
        formulas=FormulaScores(
            amstad=amstad,
            wiener_sachtextformel=wiener,
            smog_de=smog,
            lix=lix,
        ),
        parameters=params,
        rating=_rate(hix),
    )


def compute_hix_from_text(text: str, nlp: Any) -> HixResult:
    """Convenience wrapper: process text with spaCy, then compute HIX."""
    doc = nlp(text)
    return compute_hix(doc)
