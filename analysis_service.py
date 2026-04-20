"""
Leichte Sprache Analysis Service

Central service logic for analyzing texts for Leichte Sprache compliance.
Encapsulates the entire analysis pipeline from spaCy processing to result structuring.
"""

import importlib
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy

from regeln.protocol import RuleFunction

logger = logging.getLogger(__name__)


class SimpleLangAnalyzer:
    """
    Main class for analyzing texts for Leichte Sprache compliance.
    """

    def __init__(self):
        """Initialize the analyzer with spaCy model and loaded rules."""
        self.nlp = None
        self.rules = {}
        self._rule_descriptions = {
            "abkuerzungen": "BERT-basierte Abkürzungserkennung",
            "fremdwoerter": "Fremdwörter durch deutsche Alternativen ersetzen",
            "genitiv": "Genitiv-Konstruktionen vermeiden",
            "interpunktion": "Interpunktion und Satzzeichen prüfen",
            "konjunktiv": "Konjunktiv-Formen vermeiden",
            "komposita": "Lange zusammengesetzte Wörter trennen",
            "komplexitaet": "BERT-basierte Komplexitätserkennung",
            "kurze_woerter": "Zu lange Wörter (mehr als 3 Silben)",
            "mehrere_aussagen": "Mehrere Aussagen pro Satz erkennen",
            "nebensaetze": "Komplexe Nebensätze vermeiden",
            "negationen": "Verneinungen minimieren",
            "passiv_erkennung": "Passiv-Konstruktionen erkennen",
            "perplexity_saetze": "Perplexitäts-basierte Satz-Komplexitätserkennung",
            "personalpronomen": "Personalpronomen-Analyse",
            "redewendungen": "Redewendungen und Metaphern vermeiden",
            "satzlaenge": "Sätze maximal 10 Wörter lang",
            "synonyme": "Synonyme und Wortwiederholungen",
            "zahlwoerter": "BERT-basierte Zahlwörter-Erkennung (Token-Classification)",
        }

    def _load_spacy_model(self) -> bool:
        """Load the German spaCy model."""
        if self.nlp is not None:
            return True

        try:
            self.nlp = spacy.load("de_core_news_lg")
            return True
        except OSError:
            return False

    def _load_rules(self) -> int:
        """
        Dynamically load all available rule modules from the regeln/ directory.

        Returns:
            Number of successfully loaded rules
        """
        rules_path = Path(__file__).parent / "regeln"
        if not rules_path.exists():
            logger.warning("regeln/ Verzeichnis nicht gefunden: %s", rules_path)
            return 0

        # Find all rule modules (directories with regel.py)
        all_rule_dirs = [
            d
            for d in rules_path.iterdir()
            if d.is_dir() and d.name != "__pycache__" and (d / "regel.py").exists()
        ]
        expected_rules = len(all_rule_dirs)

        loaded_rules = 0
        failed_rules = []

        for rule_dir in all_rule_dirs:
            rule_name = rule_dir.name

            try:
                module = importlib.import_module(f"regeln.{rule_name}")
                if not hasattr(module, "check_rule"):
                    failed_rules.append(
                        (rule_name, "Module has no 'check_rule' function")
                    )
                    continue

                fn = module.check_rule
                if not isinstance(fn, RuleFunction):
                    failed_rules.append(
                        (rule_name, f"check_rule is not callable (got {type(fn).__name__})")
                    )
                    continue

                self.rules[rule_name] = fn
                loaded_rules += 1
            except ImportError as e:
                failed_rules.append((rule_name, f"ImportError: {str(e)}"))
            except Exception as e:
                failed_rules.append(
                    (rule_name, f"{type(e).__name__}: {str(e)}")
                )

        # Log warning if not all rules loaded
        if failed_rules:
            logger.warning(
                "%d von %d Regeln konnten nicht geladen werden",
                len(failed_rules), expected_rules,
            )
            for rule_name, error in failed_rules:
                logger.warning("  %s: %s", rule_name, error)

        if loaded_rules < expected_rules * 0.8:  # Less than 80% loaded
            logger.warning("KRITISCH: Nur %d/%d Regeln geladen!", loaded_rules, expected_rules)

        return loaded_rules

    def _extract_problematic_terms(
        self, message: str, text: str
    ) -> List[Tuple[str, int, int]]:
        """
        Extract problematic words/phrases from a rule message.

        Args:
            message: Error message from the rule
            text: Original text

        Returns:
            List of (term, start_pos, end_pos) tuples
        """
        terms = []

        if not message or not text:
            return terms

        try:
            # PRIORITY 1: Try to extract positions directly from the message
            # Format: "Text-Position: 313-316" or similar
            position_match = re.search(r"Text-Position:\s*(\d+)-(\d+)", message)
            if position_match:
                start_pos = int(position_match.group(1))
                end_pos = int(position_match.group(2))
                # Extract the term at this position from the text
                if 0 <= start_pos < len(text) and start_pos < end_pos <= len(text):
                    term = text[start_pos:end_pos]
                    return [(term, start_pos, end_pos)]

            # PRIORITY 2: Search for terms in quotes: "problematic word"
            quote_matches = re.findall(r'"([^"]+)"', message)
            for term in quote_matches:
                if not term.strip():  # Skip empty terms
                    continue

                # Escape special regex characters in the term
                escaped_term = re.escape(term)

                # Find all occurrences of the term in the text (case-insensitive, whole words)
                pattern = r"\b" + escaped_term + r"\b"
                try:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        terms.append((term, match.start(), match.end()))
                except re.error:
                    # If regex fails, use simple string search
                    start_pos = text.lower().find(term.lower())
                    if start_pos >= 0:
                        end_pos = start_pos + len(term)
                        terms.append((term, start_pos, end_pos))

        except Exception as e:
            logger.debug("Begriffe-Extraktion fehlgeschlagen: %s", e)

        return terms

    def _annotate_text(self, text: str, issues: List[Dict[str, Any]]) -> str:
        """
        Annotate the text with found violations.

        Args:
            text: Original text
            issues: List of found issues

        Returns:
            Annotated text in format: "Word[rule: word][rule2: word]"
        """
        if not issues or not text:
            return text or ""

        try:
            # Collect all annotations with their positions
            annotations = []

            for issue in issues:
                if not isinstance(issue, dict):
                    continue

                # Safe extraction of issue data
                rule_id = str(issue.get("rule_id", "unknown"))
                message = str(issue.get("message", ""))

                # Use position from the issue directly (if available)
                issue_start = issue.get("start")
                issue_end = issue.get("end")
                issue_text = str(issue.get("text", ""))

                if issue_start is not None and issue_end is not None:
                    # Use position from issue
                    if 0 <= issue_start < issue_end <= len(text):
                        annotations.append(
                            {
                                "start": issue_start,
                                "end": issue_end,
                                "regel_id": rule_id,
                                "begriff": issue_text,
                            }
                        )
                else:
                    # Fallback: search text, but only use first match
                    problematic_terms = self._extract_problematic_terms(
                        message, text
                    )
                    if problematic_terms:
                        term, start_pos, end_pos = problematic_terms[0]
                        if 0 <= start_pos < end_pos <= len(text):
                            annotations.append(
                                {
                                    "start": start_pos,
                                    "end": end_pos,
                                    "regel_id": rule_id,
                                    "begriff": term,
                                }
                            )

            if not annotations:
                return text

            # Rules that should be annotated at sentence end (refer to the whole sentence)
            sentence_level_rules = {
                "regel_mehrere_aussagen",
                "regel_nebensaetze",
                "regel_satzlaenge",
            }

            # Group annotations by end position (where they will be inserted)
            # {end_pos: [annotations]}
            annotations_by_end = {}
            for annotation in annotations:
                rule_id = annotation["regel_id"].replace("_issue", "")

                # Check if this is a sentence-level rule
                if rule_id in sentence_level_rules:
                    # Find the end of the sentence (next punctuation: . ! ? after current position)
                    original_end = annotation["end"]
                    sentence_end_pos = original_end

                    # Search for the next sentence-ending punctuation
                    for i in range(original_end, len(text)):
                        if text[i] in ".!?":
                            sentence_end_pos = i  # Position BEFORE the punctuation
                            break

                    # Use sentence end as end position
                    end_pos = sentence_end_pos
                else:
                    # Normal annotations: at their natural position
                    end_pos = annotation["end"]

                if end_pos not in annotations_by_end:
                    annotations_by_end[end_pos] = []
                annotations_by_end[end_pos].append(annotation)

            # Sort end positions in reverse (insert from back to front)
            sorted_end_positions = sorted(annotations_by_end.keys(), reverse=True)

            # Apply annotations (from back to front, to avoid shifting indices)
            annotated_text = text
            for end_pos in sorted_end_positions:
                annotations_at_pos = annotations_by_end[end_pos]

                # Create all annotation texts for this position
                annotation_texts = []
                for annotation in annotations_at_pos:
                    rule_id = annotation["regel_id"]
                    term = annotation["begriff"]

                    # Remove "_issue" suffix from rule name
                    rule_name = rule_id.replace("_issue", "")

                    # Format: [rule: term]
                    annotation_texts.append(f"[{rule_name}: {term}]")

                # Insert all annotations consecutively at the end position
                combined_annotations = "".join(annotation_texts)

                # Validate position
                if 0 <= end_pos <= len(annotated_text):
                    annotated_text = (
                        annotated_text[:end_pos]
                        + combined_annotations
                        + annotated_text[end_pos:]
                    )

            return annotated_text

        except Exception as e:
            logger.debug("Text-Annotation fehlgeschlagen: %s", e)
            return text

    def analyse_text(self, text: str) -> Dict[str, Any]:
        """
        Main function for text analysis.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with analysis results in the specified structure
        """
        if not text or not text.strip():
            return {
                "annotated_text": "",
                "statistics": {
                    "total_violations": 0,
                    "unique_violations": 0,
                    "violations_by_rule": {},
                },
                "issues": [],
            }

        # Load spaCy model
        if not self._load_spacy_model():
            return {
                "error": "SpaCy-Modell 'de_core_news_lg' konnte nicht geladen werden. "
                "Bitte installieren mit: python -m spacy download de_core_news_lg"
            }

        # Load rules
        num_rules = self._load_rules()
        if num_rules == 0:
            return {"error": "Keine Regeln gefunden im regeln/ Verzeichnis"}

        # Process text with spaCy
        doc = self.nlp(text)

        # Apply all rules
        all_violations = []
        violations_by_rule = defaultdict(int)

        for rule_name, rule_function in self.rules.items():
            try:
                rule_violations = rule_function(doc)

                if rule_violations:
                    for violation in rule_violations:
                        # Extract problematic word from the message
                        problematic_text = self._extract_problematic_text(
                            violation
                        )

                        # Extract positions from the original text
                        terms_with_positions = (
                            self._extract_problematic_terms(violation, text)
                        )

                        # Use first found position (if available)
                        start_pos = None
                        end_pos = None
                        if terms_with_positions:
                            _, start_pos, end_pos = terms_with_positions[0]

                        issue = {
                            "rule_id": f"{rule_name}_issue",
                            "text": problematic_text,
                            "message": violation,
                            "start": start_pos,
                            "end": end_pos,
                        }
                        all_violations.append(issue)
                        violations_by_rule[rule_name] += 1

            except Exception as e:
                logger.warning("Rule '%s' failed: %s: %s", rule_name, type(e).__name__, str(e))
                continue

        # Position-based deduplication to remove overlapping issues
        all_violations = self._deduplicate_by_position(all_violations)

        # Update counts after deduplication
        violations_by_rule = defaultdict(int)
        for issue in all_violations:
            rule_name = issue["rule_id"].replace("_issue", "")
            violations_by_rule[rule_name] += 1

        # Annotate text
        annotated_text = self._annotate_text(text, all_violations)

        # Count unique violations (based on rule_id + text)
        unique_violations = set()
        for issue in all_violations:
            unique_violations.add((issue["rule_id"], issue["text"]))

        # Structure result
        result = {
            "annotated_text": annotated_text,
            "statistics": {
                "total_violations": len(all_violations),
                "unique_violations": len(unique_violations),
                "violations_by_rule": dict(violations_by_rule),
            },
            "issues": all_violations,
        }

        return result

    def _deduplicate_by_position(self, issues: List[Dict]) -> List[Dict]:
        """
        Remove duplicate issues at same position, keeping most specific.

        When multiple rules flag the same text position, this method
        keeps only the most relevant issue based on rule priority.

        Args:
            issues: List of issue dictionaries

        Returns:
            Deduplicated list of issues
        """
        # Group by position
        by_position = {}
        for issue in issues:
            start = issue.get("start")
            end = issue.get("end")
            if start is not None and end is not None:
                key = (start, end)
                if key not in by_position:
                    by_position[key] = []
                by_position[key].append(issue)
            else:
                # No position - keep as-is with unique key
                by_position[id(issue)] = [issue]

        # Show ALL errors - no prioritization
        # When multiple rules flag the same text, ALL are shown
        result = []
        for pos_issues in by_position.values():
            result.extend(pos_issues)  # Keep all, not just the best

        return result

    def _extract_problematic_text(self, message: str) -> str:
        """
        Extract the problematic text from a rule message.

        Args:
            message: Rule message

        Returns:
            Problematic text or the entire message if not extractable
        """
        # Search for text in quotes
        quote_match = re.search(r'"([^"]+)"', message)
        if quote_match:
            return quote_match.group(1)

        # Fallback: first part of the message up to the first period or comma
        parts = re.split(r"[.,]", message)
        if parts:
            return parts[0].strip()[:50]  # Maximum 50 characters

        return message[:50]


# Global analyzer instance (singleton pattern for performance)
_analyzer = None


def analyse_text(text: str) -> Dict[str, Any]:
    """
    Main function for text analysis - API interface.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with analysis results:
        {
            "annotated_text": "Text with annotations",
            "statistics": {
                "total_violations": int,
                "unique_violations": int,
                "violations_by_rule": dict
            },
            "issues": [
                {
                    "rule_id": str,
                    "text": str,
                    "message": str
                }
            ]
        }
    """
    global _analyzer

    # Lazy initialization for better performance
    if _analyzer is None:
        _analyzer = SimpleLangAnalyzer()

    return _analyzer.analyse_text(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the service function
    test_text = (
        "Die komplexe Analyse wurde durchgeführt. "
        "Der Administrator evaluiert die Implementation. "
        "Das ist ein sehr langer Satz mit vielen Wörtern und Nebensätzen, "
        "der definitiv zu komplex für Leichte Sprache ist."
    )

    logger.info("=== Test der Leichte Sprache Analysis ===")
    logger.info("Text: %s", test_text)
    logger.info("Analyse-Ergebnis:")

    result = analyse_text(test_text)

    if "error" in result:
        logger.error("Fehler: %s", result["error"])
    else:
        logger.info("Statistiken:")
        stats = result["statistics"]
        logger.info("   Total Violations: %s", stats["total_violations"])
        logger.info("   Unique Violations: %s", stats["unique_violations"])
        logger.info("   By Rule: %s", stats["violations_by_rule"])

        logger.info("Issues (%d):", len(result["issues"]))
        for i, issue in enumerate(result["issues"][:5], 1):  # Only first 5
            logger.info("   %d. %s: '%s' - %s", i, issue["rule_id"], issue["text"], issue["message"])

        logger.info("Annotierter Text:")
        logger.info("   %s...", result["annotated_text"][:200])
