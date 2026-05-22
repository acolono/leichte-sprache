"""Sentence-level state machine for Stage B of /generate.

The deterministic rule analyzer emits every violation with a character
offset (`start`, `end`) into the input text. spaCy's `doc.sents` gives us
clean German sentence boundaries. Combining them, we can bucket each
violation into the sentence whose span contains it and refine sentences
one at a time — leaving already-clean sentences byte-for-byte intact.

This module is the data-plane:

  * `SentenceState`              — per-sentence record
  * `build_states`               — split text into SentenceState list
  * `assign_violations`          — bucket issues into states
  * `pick_priority`              — pop highest-weighted open sentence
  * `reassemble`                 — join states back into a single string
  * `count_edits`                — observability
  * `neighbors`                  — context for Stage B's per-sentence prompt
  * `looks_lazy`                 — aider-inspired placeholder detector
  * `similarity`                 — difflib similarity gate
  * `local_improved`             — accept/reject helper

The orchestrator in `tools/agent_optimizer.py` owns the control flow.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Aider-inspired placeholder fingerprints. The detector flags candidates that
# the LLM has cheated on instead of actually rewriting the sentence.
_LAZY_PATTERNS = [
    re.compile(r"\[\s*\.{2,}\s*\]"),           # "[...]"
    re.compile(r"\(\s*\.{2,}\s*\)"),           # "(...)"
    re.compile(r"\b(unveraendert|unverändert)\b", re.IGNORECASE),
    re.compile(r"\b(Rest\s+(bleibt|ist)\s+gleich)\b", re.IGNORECASE),
    re.compile(r"\b(wie\s+oben)\b", re.IGNORECASE),
    re.compile(r"\b(siehe\s+oben)\b", re.IGNORECASE),
]

# How short a candidate may be relative to the source sentence before we
# treat it as truncation. 0.4 picked to allow genuine compression (Leichte
# Sprache often does shorten) while still catching outright bailouts.
_MIN_LENGTH_RATIO = 0.4

# difflib similarity threshold below which we treat the candidate as not
# being a rewrite of the source at all (LLM hallucinated or grabbed the
# wrong context). Permissive on purpose; real rewrites can score very low.
_MIN_SIMILARITY = 0.15


# ---------------------------------------------------------------------------
# Per-sentence state
# ---------------------------------------------------------------------------


@dataclass
class SentenceState:
    """A single sentence's lifecycle inside the Stage B refinement loop.

    ``original`` is the sentence as it left Stage A. ``current`` starts equal
    to ``original`` and gets replaced when a candidate rewrite is accepted.
    ``violations`` is a flat list of dicts copied from the analyzer's
    ``issues`` array — only those whose character span falls inside this
    sentence.
    """

    index: int
    original: str
    current: str
    start: int                    # char offset in the Stage A output
    end: int                      # exclusive
    violations: List[Dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    status: str = "open"          # "open" | "fixed" | "abandoned"

    @property
    def weighted_violations(self) -> int:
        # Imported here to avoid a circular import with agent_optimizer.
        from tools.agent_optimizer import compute_weighted_violations

        return compute_weighted_violations(self.violations)

    @property
    def is_open(self) -> bool:
        return self.status == "open"


# ---------------------------------------------------------------------------
# Building states from a text + analysis
# ---------------------------------------------------------------------------


def build_states(text: str, nlp) -> List[SentenceState]:
    """Split ``text`` into ``SentenceState``s using the cached spaCy model.

    Sentence boundaries match what every rule module already sees via
    ``doc.sents`` — so a violation's char span is guaranteed to fall inside
    exactly one (or zero, for whitespace-only positions) sentence span.
    """
    doc = nlp(text)
    states: List[SentenceState] = []
    for idx, sent in enumerate(doc.sents):
        sentence_text = sent.text
        if not sentence_text.strip():
            continue
        states.append(
            SentenceState(
                index=idx,
                original=sentence_text,
                current=sentence_text,
                start=sent.start_char,
                end=sent.end_char,
            )
        )
    return states


def assign_violations(
    issues: List[Dict[str, Any]],
    states: List[SentenceState],
) -> None:
    """Bucket each analyzer ``issue`` into the SentenceState that contains it.

    Picks the sentence with the largest overlap when an issue's span
    crosses a boundary (rare; happens for `satzlaenge` and `perplexity` style
    text-level rules whose anchors land at sentence-end). Issues whose span
    falls outside every sentence (e.g. whitespace-only) are dropped — these
    are typically text-level summary violations the orchestrator handles
    separately at assembly time.
    """
    for state in states:
        state.violations = []

    for issue in issues:
        start = issue.get("start")
        end = issue.get("end")
        if start is None or end is None:
            continue
        best: Optional[SentenceState] = None
        best_overlap = 0
        for state in states:
            overlap_start = max(start, state.start)
            overlap_end = min(end, state.end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = state
        if best is not None:
            best.violations.append(issue)


def mark_done_if_clean(states: List[SentenceState]) -> None:
    """Flip status to ``fixed`` for any state whose violation list is empty."""
    for state in states:
        if state.status == "open" and not state.violations:
            state.status = "fixed"


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------


def has_open_problem_sentences(states: List[SentenceState]) -> bool:
    """True iff at least one open sentence still has unresolved violations."""
    return any(s.is_open and s.violations for s in states)


def pick_priority(states: List[SentenceState]) -> Optional[SentenceState]:
    """Pick the open sentence with the highest weighted violation count.

    Ties broken by lower ``attempts`` (give each sentence a chance) then by
    index (stable, leftmost first). Returns ``None`` when nothing is open.
    """
    candidates = [s for s in states if s.is_open and s.violations]
    if not candidates:
        return None
    candidates.sort(
        key=lambda s: (-s.weighted_violations, s.attempts, s.index),
    )
    return candidates[0]


def neighbors(
    states: List[SentenceState],
    index: int,
    k: int = 1,
) -> Dict[str, str]:
    """Return the up to ``k`` sentences before/after ``index`` as context.

    The orchestrator includes these in the Stage B prompt verbatim. They
    are not edited by the same call — they exist purely so the LLM can
    keep cross-sentence references (pronouns, anaphora) intact.
    """
    n = len(states)
    before = [states[i].current for i in range(max(0, index - k), index)]
    after = [
        states[i].current for i in range(index + 1, min(n, index + 1 + k))
    ]
    return {
        "before": "\n".join(before),
        "after": "\n".join(after),
    }


# ---------------------------------------------------------------------------
# Reassembly
# ---------------------------------------------------------------------------


def reassemble(states: List[SentenceState]) -> str:
    """Join sentence states into a single text.

    Stage B's per-sentence prompt produces a string that may itself contain
    multiple sentences (e.g. when ``mehrere_aussagen`` was fixed by splitting).
    Sentences are joined with single spaces; the LLM is responsible for
    terminal punctuation. Leading/trailing whitespace is stripped from each
    sentence to keep reassembly clean.
    """
    parts = [s.current.strip() for s in states if s.current.strip()]
    return " ".join(parts)


def count_edits(states: List[SentenceState]) -> int:
    """Count states whose ``current`` differs from ``original`` (after strip)."""
    return sum(1 for s in states if s.current.strip() != s.original.strip())


# ---------------------------------------------------------------------------
# Candidate validation (aider-inspired)
# ---------------------------------------------------------------------------


def looks_lazy(candidate: str, source: str) -> bool:
    """True iff ``candidate`` looks like a placeholder/truncated rewrite.

    Catches the German-prose analog of aider's "// rest unchanged" failure
    mode: explicit placeholder patterns and suspiciously short outputs.
    """
    if not candidate or not candidate.strip():
        return True
    for pattern in _LAZY_PATTERNS:
        if pattern.search(candidate):
            logger.debug("looks_lazy: matched pattern %s", pattern.pattern)
            return True
    if source and len(candidate.strip()) < _MIN_LENGTH_RATIO * len(source.strip()):
        logger.debug(
            "looks_lazy: length %d < %.0f%% of source length %d",
            len(candidate.strip()),
            _MIN_LENGTH_RATIO * 100,
            len(source.strip()),
        )
        return True
    return False


def similarity(a: str, b: str) -> float:
    """`difflib.SequenceMatcher.ratio` wrapper. 0.0 = unrelated, 1.0 = identical."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def acceptable_candidate(candidate: str, source: str) -> bool:
    """Combined gate: not lazy AND minimally similar to source."""
    if looks_lazy(candidate, source):
        return False
    return similarity(candidate, source) >= _MIN_SIMILARITY


# ---------------------------------------------------------------------------
# Acceptance: did the candidate actually improve things?
# ---------------------------------------------------------------------------


def local_improved(state: SentenceState, candidate_violations_weighted: int) -> bool:
    """True iff the candidate has strictly fewer weighted violations.

    Equal counts are rejected: when nothing changed there is no reason
    to commit a fresh LLM output. The candidate's violations are passed
    in pre-computed (the orchestrator already analyzed the trial text).
    """
    return candidate_violations_weighted < state.weighted_violations
