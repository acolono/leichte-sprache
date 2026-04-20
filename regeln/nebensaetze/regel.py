"""
Rule for detecting subordinate clauses for Leichte Sprache.
Structure-based implementation with spaCy dependency parser (NO word lists).

VERSION 2.1 - DEPENDENCY-BASED + INFINITIVE CLAUSES
Closes three critical gaps:
- Gap 1: Pure infinitive clauses (PTKZU)
- Gap 2: Attributive W-pronouns (PWAT: welches/welche/wessen)
- Gap 3: Conjunction "sodass" (dep_=cp without pos_=SCONJ)
"""

from typing import List, Set

import spacy
from spacy.tokens import Doc, Token

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401


import logging

logger = logging.getLogger(__name__)
def check_rule(doc: Doc) -> List[str]:
    """
    Detects subordinate clauses via structure-based analysis with spaCy dependency parser.

    PRINCIPLE: Uses ONLY abstract grammatical features:
    - token.dep_ (dependency label: mark, relcl, etc.)
    - token.tag_ (morphological tag: PRELS, PWS, etc.)
    - token.pos_ (part-of-speech: SCONJ, PRON, etc.)

    NO rigid word lists!

    Returns:
        List of error messages for detected subordinate clauses
    """
    errors = []
    processed_tokens: Set[int] = set()  # Verhindert doppelte Meldungen

    for i, token in enumerate(doc):
        # Skip bereits verarbeitete Tokens
        if i in processed_tokens:
            continue

        # Überspringe Zitate/Beispiele
        # Verhindert Falsch-Positive bei metalinguistischer Verwendung
        if i > 0 and i < len(doc) - 1 and doc[i - 1].is_quote and doc[i + 1].is_quote:
            continue

        # =====================================================================
        # REGEL 1: KONJUNKTIONAL-SÄTZE (Subordinierende Konjunktionen)
        # =====================================================================
        # KRITERIUM: dep_=cp (complementizer)
        # BEDEUTUNG: Im deutschen spaCy-Modell werden subordinierende Konjunktionen
        #            mit dep_=cp markiert (nicht "mark" wie im Englischen)
        # BEISPIELE: "weil", "dass", "ob", "obwohl", "als", "bevor", "um", "sodass", etc.
        # VORTEIL: Erkennt ALLE Konjunktionen ohne starre Liste
        # HINWEIS: pos_=SCONJ-Check entfernt, da "sodass" als VERB getaggt wird!
        if token.dep_ == "cp":
            errors.append(
                f'Das Wort "{token.text}" leitet einen Nebensatz ein. '
                f"Besser: Teilen Sie den Satz in zwei einfache Sätze auf."
            )
            processed_tokens.add(i)
            continue

        # =====================================================================
        # REGEL 2: RELATIVSÄTZE (Relativpronomen)
        # =====================================================================
        # KRITERIUM: tag_ in [PRELS, PRELAT] (Relativpronomen-Tags)
        # BEDEUTUNG: Erkennt "der/die/das/welche" wenn sie als Relativpronomen verwendet werden
        # BEISPIELE: "Das Haus, das groß ist...", "Die Frau, welche singt..."
        # VORTEIL: Unterscheidet Artikel von Relativpronomen automatisch
        elif token.tag_ in ["PRELS", "PRELAT"]:
            errors.append(
                f'Das Wort "{token.text}" leitet einen Relativsatz ein. '
                f"Besser: Formulieren Sie zwei einfache Sätze."
            )
            processed_tokens.add(i)
            continue

        # =====================================================================
        # REGEL 3: INDIREKTE FRAGESÄTZE & W-SÄTZE
        # =====================================================================
        # KRITERIUM: W-Pronomen/Adverbien (tag_ startet mit "PW")
        #            UND fungiert als Nebensatz-Einleiter
        # BEDEUTUNG: Erkennt "was", "wie", "wann", "wo", "warum", "welches", etc.
        # BEISPIELE:
        #   - "Ich weiß, was du tust." (PWS = substituierendes Pronomen)
        #   - "Er zeigt, wie es geht." (PWAV = adverbiales Pronomen)
        #   - "Sie fragt, wo du bist." (PWAV)
        #   - "Er weiß nicht, welches Auto..." (PWAT = attributives Pronomen)
        # HINWEIS: Hier prüfen wir auf W-Wörter, die einen eingebetteten Satz einleiten
        elif token.tag_.startswith("PW"):
            # Zusätzliche Prüfung: Ist dieses W-Wort wirklich ein Nebensatz-Einleiter?
            # Es muss ein HEAD haben, das ein Verb ist (sonst könnte es eine Hauptfrage sein)
            if _is_subclause_introducer(token):
                errors.append(
                    f'Das Wort "{token.text}" leitet einen indirekten Fragesatz/W-Satz ein. '
                    f"Besser: Formulieren Sie zwei einfache Sätze."
                )
                processed_tokens.add(i)
                continue

        # =====================================================================
        # REGEL 4: INFINITIVSÄTZE (Reine zu-Infinitive)
        # =====================================================================
        # KRITERIUM: tag_=PTKZU (Infinitiv-Partikel "zu")
        #            UND head.dep_ in satzwertigen Funktionen
        # BEDEUTUNG: Erkennt Infinitivsätze ohne einleitende Konjunktion
        # BEISPIELE:
        #   - "Er versucht, zu gehen." (head.dep_=oc)
        #   - "Die Daten zu retten, ist schwer." (head.dep_=sb)
        #   - "Sein Ziel ist, zu gewinnen." (head.dep_=sb/pd)
        # WICHTIG: Unterscheidet von Präposition "zu" (APPR)
        # HINWEIS: Infinitive mit "um", "ohne", "anstatt" werden bereits durch REGEL 1 erfasst
        elif token.tag_ == "PTKZU":
            # Hole das zugehörige Verb (HEAD des zu-Partikels)
            verb = token.head

            # Prüfe ob das Verb eine satzwertige Funktion hat
            # oc = Object Clause, sb = Subject, xcomp = Open Clausal Complement,
            # ac = Adverbial Clause, pd = Predicate, app = Apposition
            if verb.dep_ in ["oc", "sb", "xcomp", "ac", "pd", "app"]:
                errors.append(
                    f'Das Wort "{token.text}" leitet einen Infinitivsatz ein. '
                    f'Besser: Formulieren Sie zwei einfache Sätze oder verwenden Sie "dass".'
                )
                processed_tokens.add(i)
                continue

        # =====================================================================
        # REGEL 5: ASYNDETIC OBJECT CLAUSES (Without Conjunction)
        # =====================================================================
        # KRITERIUM: Finite verb with dep_="oc" (object clause) following comma
        # BEDEUTUNG: Erkennt Nebensätze ohne explizite Konjunktion
        # BEISPIELE:
        #   - "Die Menschen dachten, die Lawine kam." (kam has dep_=oc)
        #   - "Er sagte, er kommt morgen." (kommt has dep_=oc)
        # HINWEIS: These are grammatically incorrect in formal German but common
        elif token.dep_ == "oc" and token.pos_ == "VERB" and token.tag_.endswith("FIN"):
            # Check if there's a comma before this clause
            for j in range(max(0, token.i - 5), token.i):
                if doc[j].text == ",":
                    # Find the subject of this clause to report
                    clause_start = None
                    for child in token.children:
                        if child.dep_ == "sb":
                            clause_start = child.text
                            break
                    errors.append(
                        f'Nebensatz ohne Einleitewort nach Komma gefunden ("{clause_start or "..."} {token.text}"). '
                        f'Besser: Teilen Sie in zwei Sätze oder fügen Sie "dass" ein.'
                    )
                    processed_tokens.add(token.i)
                    break

    return errors


def _is_subclause_introducer(token: Token) -> bool:
    """
    Prüft, ob ein W-Wort (PW*) tatsächlich einen Nebensatz einleitet.

    KRITERIUM:
    - Das Token muss ein W-Wort sein (PWS, PWAV, PWAT, etc.)
    - Es muss eine der folgenden Dependenzen haben:
      * 'mo' (Modifier) - häufig bei eingebetteten Fragen
      * 'oc' (Object Clause) - bei W-Sätzen als Objekt (selten für W-Wort selbst)
      * 'oa' (Accusative Object) - häufig bei "was", "wen"
      * 'og' (Genitive Object) - bei "wessen"
      * 'da' (Dative) - bei "wem"
      * 'sb' (Subject) - bei W-Sätzen als Subjekt des Nebensatzes
      * 'pd' (Predicate) - selten
      * 'nk' (Noun Kernel) - bei attributiven W-Pronomen (PWAT: "welches Auto")
      * 'ag' (Genitive Attribute) - bei "wessen"
    - UND: Sein HEAD darf NICHT ROOT sein (sonst ist es eine Hauptfrage)

    BEISPIELE:
    ✓ "Ich weiß, was [oa] du tust." → True (was=oa, head=tust[oc])
    ✓ "Er zeigt, wie [mo] es geht." → True
    ✓ "Er weiß nicht, welches [nk] Auto..." → True (PWAT, head=Auto[oa]→nehmen[oc])
    ✗ "Was [oa] machst du?" → False (head=machst[ROOT] = Hauptfrage!)
    """
    # Prüfung 1: Dependency-Label deutet auf Nebensatz hin
    if token.dep_ in ["mo", "oc", "oa", "og", "da", "sb", "pd", "nk", "ag"]:
        # Zusätzliche Absicherung 1: HEAD sollte ein Verb sein
        # AUSNAHME: Bei PWAT (attributiv) ist HEAD ein Nomen!
        if token.tag_ != "PWAT" and token.head.pos_ not in ["VERB", "AUX"]:
            return False

        # Für PWAT: Prüfe ob das Nomen (HEAD) Teil eines eingebetteten Satzes ist
        if token.tag_ == "PWAT":
            noun = token.head
            # The noun must be part of an embedded clause
            # Check up to 2 levels in the dependency tree
            if noun.pos_ in ["NOUN", "PROPN"]:
                # Direct check: noun is in a subordinate role
                if noun.dep_ in ["oa", "sb", "og", "da", "nk"]:
                    # Verify the governing verb is not ROOT (would be a main question)
                    if noun.head.pos_ in ["VERB", "AUX"] and noun.head.dep_ != "ROOT":
                        return True
                # Fallback: check if any ancestor verb is subordinate
                if noun.head.pos_ in ["VERB", "AUX"]:
                    if noun.head.dep_ in ["oc", "xcomp", "ac", "mo", "rc"]:
                        return True
            return False

        # Zusätzliche Absicherung 2: HEAD darf NICHT ROOT sein
        # (sonst ist es eine Hauptfrage wie "Was machst du?")
        if token.head.dep_ == "ROOT":
            return False

        return True

    # Prüfung 2: Ist das W-Wort Teil eines durch Komma abgetrennten Nebensatzes?
    # Beispiel: "Ich weiß, was du tust." - "was" hat HEAD "tust", und das Komma davor
    if token.dep_ in ["ROOT", "oc"]:
        # Prüfe ob es ein vorheriges Komma gibt (typisch für eingebettete Sätze)
        for i in range(max(0, token.i - 3), token.i):
            if token.doc[i].text == ",":
                return True

    return False


# =========================================================================
# TESTING & VALIDATION
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import spacy

    logger.info("=" * 80)
    logger.info("NEBENSATZ-ERKENNUNG: DEPENDENCY-BASIERTER ANSATZ (v2.1)")
    logger.info("=" * 80)
    logger.info("\n KEINE Wortlisten verwendet")
    logger.info(" Nutzt NUR spaCy Dependency Parser (dep_, tag_, pos_)")
    logger.info(" NEU: Infinitivsätze (PTKZU), PWAT (welches), sodass")
    logger.info("=" * 80)

    # Testfälle
    testfaelle = [
        # Konjunktional-Sätze (REGEL 1: dep_=cp)
        "Ich bleibe zu Hause, weil es regnet.",
        "Er fragte, ob ich komme.",
        "Sie kam, obwohl sie krank war.",
        "Bevor du gehst, räume auf.",
        "Er arbeitet hart, sodass er Erfolg hat.",  # LÜCKE 3: sodass
        # Relativsätze (REGEL 2: PRELS/PRELAT)
        "Das Haus, das groß ist, steht dort.",
        "Die Frau, welche singt, ist bekannt.",
        "Der Mann, der lacht, ist mein Freund.",
        # Indirekte Fragesätze & W-Sätze (REGEL 3: PW*)
        "Ich weiß, was du tust.",
        "Er zeigt, wie es geht.",
        "Sie fragt, wo du bist.",
        "Niemand versteht, warum das passiert.",
        "Er weiß nicht, welches Auto er nehmen soll.",  # LÜCKE 2: PWAT
        "Niemand weiß, wessen Idee das war.",  # LÜCKE 2: wessen
        # Infinitivsätze (REGEL 4: PTKZU)
        "Er versucht, zu gehen.",  # LÜCKE 1
        "Die Daten zu retten, ist schwer.",  # LÜCKE 1
        "Sein Ziel ist, zu gewinnen.",  # LÜCKE 1
        # NEGATIV-Tests (sollten NICHT erkannt werden)
        "Was machst du?",  # Hauptfrage, kein Nebensatz
        "Das ist gut.",  # Kein Nebensatz
        "Ich gehe nach Hause.",  # Kein Nebensatz
        "Ich gehe zu Hause.",  # "zu" als Präposition (nicht PTKZU)
    ]

    nlp = spacy.load("de_core_news_lg")

    logger.info("\n--- TESTFÄLLE ---\n")

    for i, test_text in enumerate(testfaelle, 1):
        logger.info("[Test %s] %s", i, test_text)

        doc = nlp(test_text)
        results = check_rule(doc)

        if results:
            logger.info(" %s Nebensatz/Nebensätze gefunden:", len(results))
            for e in results:
                logger.info("  - %s", e)
        else:
            logger.info(" Keine Nebensätze erkannt")

        logger.info("")

    logger.info("=" * 80)
    logger.info("WARUM DEPENDENCY-ANALYSE ÜBERLEGEN IST:")
    logger.info("=" * 80)
    logger.info("""
1. WARTBARKEIT: Keine manuellen Wortlisten zu pflegen. Funktioniert automatisch
 mit allen Konjunktionen, die spaCy kennt.

2. ROBUSTHEIT: Erkennt mehrdeutige Wörter korrekt (z.B. "was" als Pronomen in
 "Ich weiß, was du tust" vs. Hauptfrage "Was machst du?").

3. VOLLSTÄNDIGKEIT: Findet ALLE subordinierenden Konstruktionen, auch seltene
 oder neue Konjunktionen, die nicht in einer starren Liste stehen.

4. PRÄZISION: Nutzt grammatikalische Struktur statt Wortformen. Unterscheidet
 z.B. "der" als Artikel ("der Hund") von "der" als Relativpronomen
 ("der Hund, der bellt").
 """)
    logger.info("=" * 80)
