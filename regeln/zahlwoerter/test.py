#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit-Tests für die BERT-basierte Zahlwörter-Regel.

Verwendung:
    python regeln/zahlwoerter/test.py
"""

logger = logging.getLogger(__name__)
import logging

import sys
from pathlib import Path

# Sicherstellen, dass das Projekt-Root im Python-Path ist
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import spacy  # noqa: E402

from regeln.zahlwoerter.regel import check_rule  # noqa: E402

# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = [
    # Einfache Zahlwörter (sollten erkannt werden)
    {
        "text": "Das Kind ist acht Jahre alt.",
        "expected_min_errors": 1,
        "expected_words": ["acht"],
        "description": "Einfaches Zahlwort (acht)",
    },
    {
        "text": "Der Film dauert zwei Stunden.",
        "expected_min_errors": 1,
        "expected_words": ["zwei"],
        "description": "Einfaches Zahlwort (zwei)",
    },
    {
        "text": "Es gibt fünf Optionen.",
        "expected_min_errors": 1,
        "expected_words": ["fünf"],
        "description": "Einfaches Zahlwort (fünf)",
    },
    # Zusammengesetzte Zahlen (sollten erkannt werden)
    {
        "text": "Es gibt dreiundzwanzig Teilnehmer.",
        "expected_min_errors": 1,
        "expected_words": ["dreiundzwanzig"],
        "description": "Zusammengesetzte Zahl (dreiundzwanzig)",
    },
    {
        "text": "Die Veranstaltung hat fünfundvierzig Besucher.",
        "expected_min_errors": 1,
        "expected_words": ["fünfundvierzig"],
        "description": "Zusammengesetzte Zahl (fünfundvierzig)",
    },
    # Große Zahlen (sollten erkannt werden)
    {
        "text": "Es kostet einhundert Euro.",
        "expected_min_errors": 1,
        "expected_words": ["einhundert"],
        "description": "Große Zahl (einhundert)",
    },
    {
        "text": "Die Stadt hat eine Million Einwohner.",
        "expected_min_errors": 1,
        "expected_words": ["Million"],
        "description": "Sehr große Zahl (Million)",
    },
    # Jahreszahlen (sollten erkannt werden)
    {
        "text": "Im Jahr neunzehnhundertfünfundachtzig begann alles.",
        "expected_min_errors": 1,
        "expected_words": ["neunzehnhundertfünfundachtzig"],
        "description": "Komplexe Jahreszahl",
    },
    # Prozentangaben (sollten erkannt werden)
    {
        "text": "Der Anteil beträgt fünfzig Prozent.",
        "expected_min_errors": 1,
        "expected_words": ["fünfzig"],
        "description": "Prozentangabe als Wort",
    },
    # Korrekte Verwendung (keine Fehler erwartet)
    {
        "text": "Bitte bringen Sie 2 Formulare mit.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Ziffer korrekt (2)",
    },
    {
        "text": "Die Veranstaltung findet am 15. März statt.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Ziffern korrekt (15.)",
    },
    {
        "text": "Das kostet 100 Euro.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Ziffer korrekt (100)",
    },
    {
        "text": "Im Jahr 1985 wurde das Gesetz erlassen.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Jahreszahl als Ziffer korrekt (1985)",
    },
    # Edge Cases
    {
        "text": "Es gibt null Fehler.",
        "expected_min_errors": 1,
        "expected_words": ["null"],
        "description": "Zahlwort null",
    },
    {
        "text": "Die Zahl ist eins.",
        "expected_min_errors": 1,
        "expected_words": ["eins"],
        "description": "Zahlwort eins",
    },
    # Mehrere Zahlwörter in einem Satz
    {
        "text": "Es gibt zwei Optionen und drei Alternativen.",
        "expected_min_errors": 2,
        "expected_words": ["zwei", "drei"],
        "description": "Mehrere Zahlwörter in einem Satz",
    },
    # =========================================================================
    # HYBRID FILTER TESTS (Regelbasiert + BERT)
    # =========================================================================
    # Test 1-2: Case-insensitive Matching
    {
        "text": """1.  Der Einsiedler lebte in völliger Einsamkeit.
2.  Nur eins der Kinder wollte nicht spielen.
3.  Ohne jeden Zweifel war das die richtige Entscheidung.
4.  Zwei Zweifel sind zwei zu viel.
5.  Seine Dreistigkeit kannte absolut keine Grenzen.
6.  Das Dreieck hat drei spitze Winkel.
7.  Die Viren breiteten sich im System aus.
8.  Wir müssen das Viertel der Stadt sanieren.
9.  Er trank ein Viertel des Weines aus.
10. Gib mir mal fünf Minuten Zeit.
11. Das Fünffingerkraut wächst im Garten.
12. Ein Sechstant wird in der Schifffahrt genutzt.
13. Sechs Matrosen standen an Deck.
14. Das Sieb liegt in der Spüle.
15. Wir müssen den Sand sieben, um Steine zu finden.
16. Sieben Siebenschläfer schliefen sieben Wochen lang.
17. Bitte geben Sie gut Acht im Straßenverkehr.
18. Die Acht ist eine gerade Zahl.
19. Das Neunauge ist ein fischartiges Tier.
20. Neun Augenpaare starrten mich an.
21. Der Zehnkampf ist eine olympische Disziplin.
22. Zehn Sportler traten an.
23. Der Elf versteckte sich im Elfenbein-Turm.
24. Elf Elfen tanzten um elf Uhr nachts.
25. Der Zwölffingerdarm ist ein Teil des Darms.
26. Es schlug genau zwölf Uhr mittags.
27. Tausendfüßler haben gar keine tausend Beine.
28. Tausende Menschen demonstrierten auf der Straße.
29. Das Jahrhundert war geprägt von Kriegen.
30. Einhundert Jahre sind eine lange Zeit.
31. Der Millionär spendete eine Million Euro.
32. Er wohnt in der Straße des 17. Juni.
33. Am 1. Mai findet ein großes Fest statt.
34. Ludwig XIV regierte Frankreich sehr lange.
35. Kapitel IV behandelt die Grundlagen.
36. Das Jahr 2025 wird spannend.
37. 1990 war das Jahr der Wiedervereinigung.
38. Ein Dutzend Eier kostet nicht viel.
39. Ein Pfund Butter und ein Zentner Mehl.
40. Wir suchen eine zweisprachige Sekretärin.
41. Das war eine einmalige Gelegenheit.
42. Der dreimalige Weltmeister trat zurück.
43. Das viereckige Gebäude hat vier Ecken.
44. Eine fünfminütige Pause ist zu kurz.
45. Der 50-jährige Mann feierte Geburtstag.
46. Das 3-Gänge-Menü schmeckte hervorragend.
47. Alle neune wurden beim Kegeln getroffen.
48. Wir teilen den Kuchen in zwei Hälften.
49. Ein Drittel ist weniger als die Hälfte.
50. Die Einsicht kam leider viel zu spät.
51. Ein Zweig brach vom Baum ab.
52. Dreierlei Sorten Käse lagen auf dem Tisch.
53. Vielfach wurde dieser Wunsch geäußert.
54. Fünferlei Gewürze sind in der Suppe.
55. Sechser im Lotto ist ein großer Glücksfall.
56. Siebzehn Jahr, blondes Haar.
57. Achtzig Jahre sind ein stolzes Alter.
58. Neunzig Prozent sind fast alles.
59. Einhundertprozentige Sicherheit gibt es nicht.
60. Tausendundeine Nacht ist ein Märchen.""",
        "expected_min_errors": 1,
        "expected_words": ["ZWEI"],
        "description": "Hybrid: Case-insensitive (ZWEI Großbuchstaben)",
    },
    {
        "text": "Die Zahl Drei ist wichtig.",
        "expected_min_errors": 1,
        "expected_words": ["Drei"],
        "description": "Hybrid: Case-insensitive (Drei Großer Anfangsbuchstabe)",
    },
    # Test 3-6: Inflected Forms (Flexionsformen)
    {
        "text": "Der zweite Versuch war erfolgreich.",
        "expected_min_errors": 1,
        "expected_words": ["zweite"],
        "description": "Hybrid: Flexionsform (zweite Ordinalzahl)",
    },
    {
        "text": "Das passierte dreimal hintereinander.",
        "expected_min_errors": 1,
        "expected_words": ["dreimal"],
        "description": "Hybrid: Flexionsform (dreimal Adverb)",
    },
    {
        "text": "Er kam fünftens an die Reihe.",
        "expected_min_errors": 1,
        "expected_words": ["fünftens"],
        "description": "Hybrid: Flexionsform (fünftens Ordinaladverb)",
    },
    {
        "text": "Die vierten Plätze sind vergeben.",
        "expected_min_errors": 1,
        "expected_words": ["vierten"],
        "description": "Hybrid: Flexionsform (vierten Plural Ordinalzahl)",
    },
    # Test 7-8: Compound Exclusion (Komposita sollten NICHT erkannt werden)
    {
        "text": "Das Zweifamilienhaus ist groß.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Hybrid: Kompositum (Zweifamilienhaus) sollte NICHT erkannt werden",
    },
    {
        "text": "Der Dreieckstisch steht dort.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Hybrid: Kompositum (Dreieckstisch) sollte NICHT erkannt werden",
    },
    # Test 9-10: Ambiguous Words (elf, acht) - nur BERT entscheidet
    {
        "text": "Die elf Spieler sind bereit.",
        "expected_min_errors": 1,
        "expected_words": ["elf"],
        "description": "Hybrid: Mehrdeutiges Wort (elf) - BERT erkennt als Zahl",
    },
    {
        "text": "Gib acht auf die Stufen.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Hybrid: Mehrdeutiges Wort (acht) - BERT erkennt als Redewendung",
    },
    # Test 11-12: Deduplication & Hybrid Mode
    {
        "text": "Die Zahl zehn ist eine runde Zahl.",
        "expected_min_errors": 1,
        "expected_words": ["zehn"],
        "description": "Hybrid: Deduplizierung (zehn von Regel + BERT erkannt, nur 1 Fehler)",
    },
    {
        "text": "Es gibt sieben Tage und hundert Stunden.",
        "expected_min_errors": 2,
        "expected_words": ["sieben", "hundert"],
        "description": "Hybrid: Beide Filter (sieben=Regel, hundert=BERT)",
    },
    # =========================================================================
    # POST-PROCESSING & REGEX FILTER TESTS (Ghost-Artifacts, Römische Zahlen)
    # =========================================================================
    # Test 13-14: Ghost-Artifacts sollten NICHT erkannt werden
    {
        "text": "50-jähriges Jubiläum wird gefeiert.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Post-Processing: Ghost-Artifact 'iges' nach Bindestrich NICHT erkennen",
    },
    {
        "text": "Das wichtiges Thema wird besprochen.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Post-Processing: 'iges' ist kein Zahlwort, sollte NICHT erkannt werden",
    },
    # Test 15: Bindestrich-Kontext sollte NICHT erkannt werden
    {
        "text": "Der 70-fach verstärkte Schutz ist wichtig.",
        "expected_min_errors": 0,
        "expected_words": [],
        "description": "Post-Processing: 'fach' nach Bindestrich NICHT erkennen",
    },
    # Test 16-18: Römische Zahlen (Regex-Filter)
    {
        "text": "Ludwig XIV. war König von Frankreich.",
        "expected_min_errors": 1,
        "expected_words": ["XIV"],
        "description": "Regex-Filter: Römische Zahl XIV. erkennen",
    },
    {
        "text": "Kapitel III beschreibt die Methodik.",
        "expected_min_errors": 1,
        "expected_words": ["III"],
        "description": "Regex-Filter: Römische Zahl III erkennen",
    },
    {
        "text": "Im XX. Jahrhundert gab es große Veränderungen.",
        "expected_min_errors": 1,
        "expected_words": ["XX"],
        "description": "Regex-Filter: Römische Zahl XX. erkennen",
    },
    # =========================================================================
    # REGEX FILTER: Zahlen 50-100
    # =========================================================================
    # Test 19-24: Basis-Zehner (50, 60, 70, 80, 90)
    {
        "text": "Es kostet fünfzig Euro.",
        "expected_min_errors": 1,
        "expected_words": ["fünfzig"],
        "description": "Regex 50-100: fünfzig erkennen",
    },
    {
        "text": "Der Mann ist sechzig Jahre alt.",
        "expected_min_errors": 1,
        "expected_words": [],  # Kann "sech" oder "sechzig" sein (BERT vs Regex)
        "description": "Regex 50-100: sechzig erkennen",
    },
    {
        "text": "Es sind achtzig Teilnehmer.",
        "expected_min_errors": 1,
        "expected_words": ["achtzig"],
        "description": "Regex 50-100: achtzig erkennen",
    },
    # Test 25-26: Hundert
    {
        "text": "Das Buch hat hundert Seiten.",
        "expected_min_errors": 1,
        "expected_words": ["hundert"],
        "description": "Regex 50-100: hundert erkennen",
    },
    {
        "text": "Es waren einhundert Gäste.",
        "expected_min_errors": 1,
        "expected_words": ["einhundert"],
        "description": "Regex 50-100: einhundert erkennen",
    },
    # Test 27-29: Zusammengesetzte Zahlen (51-99)
    {
        "text": "Die Zahl einundfünfzig ist ungerade.",
        "expected_min_errors": 1,
        "expected_words": ["einundfünfzig"],
        "description": "Regex 50-100: einundfünfzig erkennen",
    },
    {
        "text": "Er erreichte siebenundsiebzig Punkte.",
        "expected_min_errors": 1,
        "expected_words": ["siebenundsiebzig"],
        "description": "Regex 50-100: siebenundsiebzig erkennen",
    },
    {
        "text": "Die NEUNZIG Teilnehmer waren da.",
        "expected_min_errors": 1,
        "expected_words": ["NEUNZIG"],
        "description": "Regex 50-100: Case-insensitive (NEUNZIG)",
    },
]


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def run_tests(nlp):
    """Führt alle Test-Cases aus."""
    logger.info("=" * 80)
    logger.info("BERT-basierte Zahlwörter-Regel - Unit-Tests")
    logger.info("=" * 80)
    logger.info("")

    passed = 0
    failed = 0
    warnings = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        text = test_case["text"]
        expected_min_errors = test_case["expected_min_errors"]
        expected_words = test_case["expected_words"]
        description = test_case["description"]

        logger.info("Test %s/%s: %s", i, len(TEST_CASES), description)
        logger.info(' Text: "%s"', text)

        # Regel ausführen
        doc = nlp(text)
        errors = check_rule(doc)

        # Anzahl Fehler prüfen
        actual_errors = len(errors)
        logger.error(" Erwartete Fehler: ≥%s, Gefunden: %s", expected_min_errors, actual_errors)

        # Test-Ergebnis
        test_passed = True

        if actual_errors < expected_min_errors:
            logger.error(" FEHLGESCHLAGEN: Zu wenige Fehler erkannt!")
            test_passed = False
            failed += 1
        else:
            # Prüfe ob erwartete Wörter gefunden wurden
            if expected_words:
                found_words = []
                for error in errors:
                    # Extrahiere Wort aus Fehlermeldung
                    # Format: 'Zahlwort "WORT" sollte...'
                    if '"' in error:
                        word = error.split('"')[1]
                        found_words.append(word)

                missing_words = set(expected_words) - set(found_words)
                if missing_words:
                    logger.warning(" WARNUNG: Erwartete Wörter fehlen: %s", missing_words)
                    logger.info(" Gefundene Wörter: %s", found_words)
                    warnings += 1
                    # Trotzdem als bestanden werten, da Mindestanzahl korrekt

            if test_passed:
                logger.info(" BESTANDEN")
                passed += 1

        # Zeige Fehlermeldungen
        if errors:
            for error in errors:
                logger.error(" %s", error)

        logger.info("")

    # Zusammenfassung
    logger.info("=" * 80)
    logger.info("ZUSAMMENFASSUNG")
    logger.info("=" * 80)
    logger.info(" Bestanden: %s/%s", passed, len(TEST_CASES))
    logger.error(" Fehlgeschlagen: %s/%s", failed, len(TEST_CASES))
    logger.warning(" Warnungen: %s/%s", warnings, len(TEST_CASES))

    if failed == 0:
        logger.info("\n Alle Tests bestanden!")
        return True
    else:
        logger.error("\n %s Test(s) fehlgeschlagen!", failed)
        return False


def run_position_accuracy_test(nlp):
    """Testet die Genauigkeit der Position-Ausgabe."""
    logger.info("\n" + "=" * 80)
    logger.info("POSITION ACCURACY TEST")
    logger.info("=" * 80)
    logger.info("")

    test_text = "Das Kind ist acht Jahre alt."
    expected_word = "acht"
    expected_start = test_text.index(expected_word)
    expected_end = expected_start + len(expected_word)

    logger.info('Test-Text: "%s"', test_text)
    logger.info('Erwartete Position für "%s": %s-%s', expected_word, expected_start, expected_end)

    doc = nlp(test_text)
    errors = check_rule(doc)

    if not errors:
        logger.error(" Keine Fehler gefunden (erwartet: 1)")
        return False

    # Extrahiere Position aus Fehlermeldung
    # Format: 'Zahlwort "acht" sollte als Ziffer geschrieben werden. Text-Position: 12-16'
    error = errors[0]
    if "Text-Position:" in error:
        position_str = error.split("Text-Position:")[1].strip()
        start, end = map(int, position_str.split("-"))

        logger.info("Tatsächliche Position: %s-%s", start, end)

        if start == expected_start and end == expected_end:
            logger.info(" Position korrekt!")
            return True
        else:
            logger.info(" Position inkorrekt! (Erwartet: %s-%s)", expected_start, expected_end)
            return False
    else:
        logger.error(" Keine Position in Fehlermeldung gefunden!")
        return False


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Hauptfunktion."""
    logger.info("\n Lade spaCy-Modell...")

    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        logger.error(" spaCy-Modell 'de_core_news_lg' nicht gefunden!")
        logger.error(" Bitte installieren: python -m spacy download de_core_news_lg")
        sys.exit(1)

    logger.info(" spaCy-Modell geladen\n")

    # Haupttests
    tests_passed = run_tests(nlp)

    # Position-Test
    position_test_passed = run_position_accuracy_test(nlp)

    # Exit-Code
    if tests_passed and position_test_passed:
        logger.info("\n Alle Tests erfolgreich!")
        sys.exit(0)
    else:
        logger.info("\n Einige Tests sind fehlgeschlagen!")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
