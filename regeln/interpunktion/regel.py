"""
WHITELIST-BASED detection of punctuation issues in Leichte Sprache (v3.0).

FUNDAMENTAL CHANGE in v3.0:
- PARADIGM SHIFT: From blacklist to WHITELIST
- ONLY allowed characters: Period (.), ?, !, :, quotation marks (typographic), mediopunkt (middot)
- FORBIDDEN: Straight quotation marks ("), all other special characters
- ALL other characters are detected as errors
- Extremely thorough detection of ALL special characters, symbols, Unicode characters

LEICHTE SPRACHE RULE:
"Zur Verfuegung stehen Punkt, Frage-, Ausrufezeichen, Doppelpunkt, Anfuehrungszeichen, Mediopunkt."

IMPORTANT:
- Colon (:) is ALWAYS allowed (not context-dependent)
- Straight quotation marks (") are FORBIDDEN -> Use typographic ones
"""

import re
import unicodedata
from typing import Dict, List

import spacy
from spacy.tokens import Doc, Token

# Import configuration
from .config import COMMON_SHORT_WORDS

# =========================================================================
# WHITELIST: NUR DIESE ZEICHEN SIND ERLAUBT
# =========================================================================

# ALWAYS allowed punctuation marks (no context check needed)
import logging

logger = logging.getLogger(__name__)
ALLOWED_PUNCTUATION_MARKS = {
    ".": "Punkt",
    "?": "Fragezeichen",
    "!": "Ausrufezeichen",
    ":": "Doppelpunkt",  # GEÄNDERT: Doppelpunkt ist IMMER erlaubt
    "·": "Mediopunkt",
    "•": "Aufzählungspunkt (Mediopunkt-Variante)",  # Häufig verwendete Alternative
}

# Anführungszeichen (NUR typografische Varianten erlaubt, GERADE " VERBOTEN!)
ALLOWED_QUOTATION_MARKS = {
    # WICHTIG: KEIN gerades " (U+0022) - NUR typografische Zeichen!
    "\u201c": "Typografische Anführungszeichen (öffnend)",  # "
    "\u201d": "Typografische Anführungszeichen (schließend)",  # "
    "\u201e": "Deutsche Anführungszeichen (öffnend)",  # „
    "\u201a": "Einfache deutsche Anführungszeichen",  # ‚
    "\u2018": "Einfache typografische Anführungszeichen (öffnend)",  # '
    "\u2019": "Einfache typografische Anführungszeichen (schließend)",  # '
    "\u00bb": "Französische Guillemets (öffnend)",  # »
    "\u00ab": "Französische Guillemets (schließend)",  # «
    "\u203a": "Einfache Guillemets (öffnend)",  # ›
    "\u2039": "Einfache Guillemets (schließend)",  # ‹
}

# Kontextabhängig erlaubte Zeichen
CONTEXT_ALLOWED_CHARACTERS = {
    ",": "Komma",  # Bis zu 2 pro Satz erlaubt
    # ENTFERNT: ":" (Doppelpunkt) - ist jetzt IMMER erlaubt (siehe ALLOWED_PUNCTUATION_MARKS)
}

# Whitespace und technische Zeichen (IMMER ignorieren, nie als Fehler melden)
TECHNICAL_CHARACTERS = {
    " ",
    "\t",
    "\n",
    "\r",
    "\u00a0",  # Whitespace
    "\u200b",
    "\u200c",
    "\u200d",  # Zero-width characters
    "\ufeff",  # Byte order mark
}

# KOMBINIERE ALLE ERLAUBTEN ZEICHEN
ALL_ALLOWED_CHARACTERS = (
    set(ALLOWED_PUNCTUATION_MARKS.keys())
    | set(ALLOWED_QUOTATION_MARKS.keys())
    | set(CONTEXT_ALLOWED_CHARACTERS.keys())
    | TECHNICAL_CHARACTERS
)

# Kontexte wo Doppelpunkt/Komma AKZEPTABEL sind
ACCEPTABLE_CONTEXTS = {
    "zeit": ["uhr", "h", "stunde", "minute", "sekunde"],  # 14:30 Uhr
    "liste": [
        "punkte",
        "punkt",
        "schritte",
        "schritt",
        "regeln",
        "regel",
        "beispiele",
        "beispiel",
        "folgende",
        "sind",
        "genannt",
        "lauten",
        "teams",
        "team",
    ],
    "definition": ["das ist", "bedeutet", "heißt", "erklärt"],  # Das heißt:
    "zitat": ["sagte", "meinte", "erklärte", "antwortete"],  # Er sagte:
}

MAX_COMMAS_PER_SENTENCE = 2  # Maximum 2 commas per sentence in Leichte Sprache

# =========================================================================
# HELPER FUNCTIONS: ZEICHEN-KLASSIFIZIERUNG
# =========================================================================


def is_letter_or_digit(zeichen: str) -> bool:
    """Checks if a character is a letter or digit."""
    if not zeichen:
        return False
    return zeichen.isalpha() or zeichen.isdigit()


def is_allowed_character(zeichen: str) -> bool:
    """Checks if a character is in the whitelist."""
    return zeichen in ALL_ALLOWED_CHARACTERS


def classify_forbidden_character(zeichen: str) -> Dict[str, str]:
    """
    Classifies a forbidden character and returns detailed information.

    Extremely thorough: detects all Unicode categories of special characters.
    """
    # Unicode-Kategorie ermitteln
    try:
        kategorie = unicodedata.category(zeichen)
        name = unicodedata.name(zeichen, "UNBEKANNTES ZEICHEN")
    except (ValueError, TypeError):
        kategorie = "UNKNOWN"
        name = "UNBEKANNTES ZEICHEN"

    # Mapping: Häufige Zeichen mit deutschen Namen
    bekannte_zeichen = {
        # Mathematische Symbole
        "+": ("Plus-Zeichen", 'Schreiben Sie "plus" aus.'),
        "−": ("Minus-Zeichen", 'Schreiben Sie "minus" aus.'),
        "-": ("Bindestrich/Minus", "Verwenden Sie einen Punkt statt Bindestrich."),
        "×": ("Malzeichen", 'Schreiben Sie "mal" aus.'),
        "÷": ("Geteilt-Zeichen", 'Schreiben Sie "geteilt durch" aus.'),
        "=": ("Gleichheitszeichen", 'Schreiben Sie "ist gleich" aus.'),
        "≈": ("Ungefähr-gleich", 'Schreiben Sie "ungefähr" aus.'),
        "≠": ("Ungleich", 'Schreiben Sie "ist nicht gleich" aus.'),
        "<": ("Kleiner-als", 'Schreiben Sie "kleiner als" aus.'),
        ">": ("Größer-als", 'Schreiben Sie "größer als" aus.'),
        "≤": ("Kleiner-gleich", 'Schreiben Sie "kleiner oder gleich" aus.'),
        "≥": ("Größer-gleich", 'Schreiben Sie "größer oder gleich" aus.'),
        "%": ("Prozent-Zeichen", 'Schreiben Sie "Prozent" aus.'),
        "‰": ("Promille-Zeichen", 'Schreiben Sie "Promille" aus.'),
        # Währungen
        "$": ("Dollar-Zeichen", "Vermeiden Sie Währungssymbole im Fließtext."),
        "€": ("Euro-Zeichen", "Schreiben Sie Euro aus."),
        "£": ("Pfund-Zeichen", "Schreiben Sie Pfund aus."),
        "¥": ("Yen-Zeichen", "Schreiben Sie Yen aus."),
        "₹": ("Rupie-Zeichen", "Schreiben Sie Rupie aus."),
        "₽": ("Rubel-Zeichen", "Schreiben Sie Rubel aus."),
        "₩": ("Won-Zeichen", "Schreiben Sie Won aus."),
        "₦": ("Naira-Zeichen", "Schreiben Sie Naira aus."),
        "₪": ("Schekel-Zeichen", "Schreiben Sie Schekel aus."),
        # Logische/technische Symbole
        "&": ("Und-Zeichen (Ampersand)", 'Schreiben Sie "und" aus.'),
        "@": ("At-Zeichen", "Vermeiden Sie E-Mail-Symbole im Fließtext."),
        "#": ("Hashtag/Raute", "Vermeiden Sie Social-Media-Symbole im Fließtext."),
        "*": (
            "Sternchen/Asterisk",
            "Vermeiden Sie Sternchen (oft für Gendersprache verwendet).",
        ),
        "^": ("Zirkumflex", "Vermeiden Sie technische Symbole."),
        "~": ("Tilde", "Vermeiden Sie technische Symbole."),
        "`": ("Gravis", "Vermeiden Sie technische Symbole."),
        "|": ("Pipe/Vertikaler Strich", 'Schreiben Sie "oder" aus.'),
        "\\": ("Backslash", "Vermeiden Sie technische Symbole."),
        "/": ("Schrägstrich", 'Schreiben Sie "oder" aus.'),
        # Klammern (ALLE verboten in Leichter Sprache)
        "(": (
            "Runde Klammer (öffnend)",
            "Schreiben Sie Informationen in einem eigenen Satz.",
        ),
        ")": (
            "Runde Klammer (schließend)",
            "Schreiben Sie Informationen in einem eigenen Satz.",
        ),
        "[": ("Eckige Klammer (öffnend)", "Vermeiden Sie Klammern."),
        "]": ("Eckige Klammer (schließend)", "Vermeiden Sie Klammern."),
        "{": ("Geschweifte Klammer (öffnend)", "Vermeiden Sie Klammern."),
        "}": ("Geschweifte Klammer (schließend)", "Vermeiden Sie Klammern."),
        "⟨": ("Spitze Klammer (öffnend)", "Vermeiden Sie Klammern."),
        "⟩": ("Spitze Klammer (schließend)", "Vermeiden Sie Klammern."),
        # Gedankenstriche und Bindestriche
        "–": (
            "Gedankenstrich (en-dash)",
            "Verwenden Sie einen Punkt und beginnen Sie einen neuen Satz.",
        ),
        "—": (
            "Langer Gedankenstrich (em-dash)",
            "Verwenden Sie einen Punkt und beginnen Sie einen neuen Satz.",
        ),
        "‒": ("Ziffernbreiter Strich", "Verwenden Sie einen Punkt."),
        "―": ("Horizontaler Strich", "Verwenden Sie einen Punkt."),
        # Anführungszeichen (GERADE ANFÜHRUNGSZEICHEN VERBOTEN)
        '"': (
            "Gerade Anführungszeichen",
            "Verwenden Sie typografische Anführungszeichen (\u201c \u201d \u201e etc.).",
        ),
        "'": (
            "Einfaches Anführungszeichen/Apostroph",
            "Verwenden Sie typografische Anführungszeichen.",
        ),
        # Auslassungspunkte
        "…": ("Ellipse (Auslassungspunkte)", "Schreiben Sie Gedanken vollständig aus."),
        # Semikolon
        ";": (
            "Semikolon/Strichpunkt",
            "Verwenden Sie einen Punkt und beginnen Sie einen neuen Satz.",
        ),
        # Paragraphen und rechtliche Symbole
        "§": (
            "Paragraph-Zeichen",
            "Schreiben Sie Paragraphen aus (z.B. 'Paragraph 5').",
        ),
        "©": ("Copyright-Zeichen", "Schreiben Sie 'Copyright' aus."),
        "®": ("Registered-Trademark", "Vermeiden Sie Markenzeichen im Fließtext."),
        "™": ("Trademark-Zeichen", "Vermeiden Sie Markenzeichen im Fließtext."),
        # Pfeile
        "→": ("Pfeil nach rechts", 'Schreiben Sie "zu" oder "führt zu" aus.'),
        "←": ("Pfeil nach links", 'Schreiben Sie "von" oder "zurück zu" aus.'),
        "↑": ("Pfeil nach oben", 'Schreiben Sie "hoch" oder "steigt" aus.'),
        "↓": ("Pfeil nach unten", 'Schreiben Sie "runter" oder "sinkt" aus.'),
        "↔": ("Doppelpfeil", 'Schreiben Sie "hin und her" aus.'),
        "⇒": ("Dicker Pfeil rechts", 'Schreiben Sie "bedeutet" oder "führt zu" aus.'),
        "⇐": ("Dicker Pfeil links", 'Schreiben Sie "folgt aus" aus.'),
        # Sterne und Symbole
        "★": ("Gefüllter Stern", "Vermeiden Sie Symbole."),
        "☆": ("Leerer Stern", "Vermeiden Sie Symbole."),
        "♥": ("Herz", "Vermeiden Sie Symbole."),
        "♦": ("Raute", "Vermeiden Sie Symbole."),
        "♣": ("Kreuz", "Vermeiden Sie Symbole."),
        "♠": ("Pik", "Vermeiden Sie Symbole."),
        # Weitere häufige Symbole
        "°": ("Grad-Zeichen", "Schreiben Sie 'Grad' aus."),
        "′": ("Prime/Minuten", "Schreiben Sie 'Minuten' aus."),
        "″": ("Doppel-Prime/Sekunden", "Schreiben Sie 'Sekunden' aus."),
        "†": ("Kreuz/Dagger", "Vermeiden Sie Symbole."),
        "‡": ("Doppelkreuz", "Vermeiden Sie Symbole."),
        "¶": ("Absatzzeichen", "Vermeiden Sie technische Symbole."),
        "¡": ("Umgekehrtes Ausrufezeichen", "Verwenden Sie normales Ausrufezeichen."),
        "¿": ("Umgekehrtes Fragezeichen", "Verwenden Sie normales Fragezeichen."),
        # Mathematische/wissenschaftliche Symbole
        "±": ("Plus-Minus", 'Schreiben Sie "plus oder minus" aus.'),
        "∓": ("Minus-Plus", 'Schreiben Sie "minus oder plus" aus.'),
        "∞": ("Unendlich", 'Schreiben Sie "unendlich" aus.'),
        "√": ("Quadratwurzel", 'Schreiben Sie "Wurzel aus" aus.'),
        "∑": ("Summe", 'Schreiben Sie "Summe" aus.'),
        "∏": ("Produkt", 'Schreiben Sie "Produkt" aus.'),
        "∫": ("Integral", "Vermeiden Sie mathematische Symbole."),
        "∂": ("Partiell", "Vermeiden Sie mathematische Symbole."),
        "∇": ("Nabla", "Vermeiden Sie mathematische Symbole."),
        # Mengenlehre
        "∈": ("Element von", 'Schreiben Sie "ist in" aus.'),
        "∉": ("Nicht Element von", 'Schreiben Sie "ist nicht in" aus.'),
        "∪": ("Vereinigung", 'Schreiben Sie "oder" aus.'),
        "∩": ("Schnittmenge", 'Schreiben Sie "und" aus.'),
        "⊂": ("Teilmenge", 'Schreiben Sie "ist Teil von" aus.'),
        "⊃": ("Obermenge", 'Schreiben Sie "enthält" aus.'),
        "∅": ("Leere Menge", 'Schreiben Sie "leer" aus.'),
    }

    if zeichen in bekannte_zeichen:
        bezeichnung, vorschlag = bekannte_zeichen[zeichen]
        return {
            "zeichen": zeichen,
            "bezeichnung": bezeichnung,
            "unicode_name": name,
            "kategorie": kategorie,
            "vorschlag": vorschlag,
        }

    # Fallback für unbekannte Zeichen basierend auf Unicode-Kategorie
    kategorie_mapping = {
        "Sm": ("Mathematisches Symbol", "Vermeiden Sie mathematische Symbole."),
        "Sc": ("Währungssymbol", "Schreiben Sie Währungen aus."),
        "Sk": ("Modifier-Symbol", "Vermeiden Sie technische Symbole."),
        "So": ("Sonstiges Symbol", "Vermeiden Sie Symbole."),
        "Ps": (
            "Öffnende Klammer",
            "Schreiben Sie Informationen in einem eigenen Satz.",
        ),
        "Pe": (
            "Schließende Klammer",
            "Schreiben Sie Informationen in einem eigenen Satz.",
        ),
        "Pd": ("Strich/Bindestrich", "Verwenden Sie einen Punkt."),
        "Po": ("Andere Interpunktion", "Verwenden Sie einfache Satzzeichen."),
        "Pi": (
            "Öffnendes Anführungszeichen",
            "Verwenden Sie deutsche Anführungszeichen.",
        ),
        "Pf": (
            "Schließendes Anführungszeichen",
            "Verwenden Sie deutsche Anführungszeichen.",
        ),
    }

    if kategorie in kategorie_mapping:
        bezeichnung, vorschlag = kategorie_mapping[kategorie]
    else:
        bezeichnung = f"Sonderzeichen ({kategorie})"
        vorschlag = "Vermeiden Sie komplexe Zeichen. Verwenden Sie einfache Sprache."

    return {
        "zeichen": zeichen,
        "bezeichnung": bezeichnung,
        "unicode_name": name,
        "kategorie": kategorie,
        "vorschlag": vorschlag,
    }


# =========================================================================
# KONTEXT-PRÜFUNG FÜR DOPPELPUNKT UND KOMMA
# =========================================================================


def is_context_acceptable(token: Token) -> bool:
    """
    Checks if a context-dependent character (: or ,) is acceptable.
    """
    zeichen = token.text
    doc = token.doc

    # --- DOPPELPUNKT : ---
    if zeichen == ":":
        sent_start = token.sent.start

        # Sammle 5 Wörter VOR dem Doppelpunkt (index-basiert)
        words_before = []
        for i in range(max(sent_start, token.i - 10), token.i):
            t = doc[i]
            if not t.is_punct and not t.is_space:
                words_before.append(t.text.lower())
                if len(words_before) >= 5:
                    break
        words_before = words_before[-5:]  # Nur letzte 5

        # Sammle 5 Wörter NACH dem Doppelpunkt
        sent_end = token.sent.end
        words_after = []
        for i in range(token.i + 1, min(sent_end, token.i + 11)):
            t = doc[i]
            if not t.is_punct and not t.is_space:
                words_after.append(t.text.lower())
                if len(words_after) >= 5:
                    break

        context_words = words_before + words_after

        # Zeit: "14:30", "um 15:00 Uhr"
        if any(word in ACCEPTABLE_CONTEXTS["zeit"] for word in context_words):
            return True

        # Listen/Aufzählungen: "Folgende Teams sind involviert:"
        if any(word in ACCEPTABLE_CONTEXTS["liste"] for word in words_before):
            return True

        # Definitionen: "Das bedeutet:", "Das heißt:"
        sent_text = token.sent.text.lower()
        if any(phrase in sent_text for phrase in ACCEPTABLE_CONTEXTS["definition"]):
            return True

        # Zitate: "Er sagte:", "Sie meinte:"
        if any(word in ACCEPTABLE_CONTEXTS["zitat"] for word in context_words):
            return True

    # --- KOMMA , ---
    # Kommas sind akzeptabel, aber maximal 2 pro Satz
    # Die Prüfung erfolgt in analyze_comma_structures()

    return False


# =========================================================================
# KOMMA-STRUKTUR-ANALYSE
# =========================================================================


def analyze_comma_structures(doc: Doc) -> List[str]:
    """
    Analyzes comma structures via dependency parsing.
    """
    issues = []

    for i, sent in enumerate(doc.sents, 1):
        commas = [token for token in sent if token.text == ","]

        if len(commas) > MAX_COMMAS_PER_SENTENCE:
            # Analyze comma functions
            comma_functions = []
            for comma in commas:
                if comma.head.pos_ == "SCONJ":
                    comma_functions.append("Nebensatz")
                elif any(child.pos_ == "CCONJ" for child in comma.head.children):
                    comma_functions.append("Aufzählung")
                elif comma.head.pos_ in ["NOUN", "PROPN"]:
                    comma_functions.append("Zusatzinfo")
                else:
                    comma_functions.append("Trennung")

            issues.append(
                f"Satz {i} enthält {len(commas)} Kommas ({', '.join(set(comma_functions))}). "
                f"Besser: Teilen Sie in {len(commas) + 1} separate Sätze auf. "
                f"Leichte Sprache erlaubt maximal {MAX_COMMAS_PER_SENTENCE} Kommas pro Satz."
            )

    return issues


# =========================================================================
# ABKÜRZUNGS-ERKENNUNG
# =========================================================================


def detect_abbreviations(doc: Doc) -> List[str]:
    """
    Detects abbreviations (forbidden in Leichte Sprache).

    Finds: Dr., z.B., usw., etc., ca., inkl., ggf., bzw.
    """
    errors = []
    found = set()

    for token in doc:
        # Method 1: Period as separate token, but NOT end of sentence
        if token.text == "." and not token.is_sent_end and token.i > 0:
            prev_token = doc[token.i - 1]

            # Plausible abbreviation stems:
            # - Known abbreviations: usw, etc, inkl, ggf, bzw
            # - OR very short words (<=3 chars) that are NOT in the whitelist
            # BUT: Single uppercase letters at sentence end (like "Team B.") are NOT abbreviations
            # BUT: Common short German words (ist, ab, an, etc.) are NOT abbreviations
            known_abbreviations = {
                "usw",
                "etc",
                "inkl",
                "ggf",
                "bzw",
                "evtl",
                "max",
                "min",
                "ca",
                "nr",
                "tel",
                "str",
            }
            is_known_abbreviation = prev_token.text.lower() in known_abbreviations
            is_short_word = len(prev_token.text) <= 3 and not (
                len(prev_token.text) == 1 and token.is_sent_end
            )
            is_common_german_word = (
                prev_token.text.lower() in COMMON_SHORT_WORDS
            )

            is_plausible_stem = prev_token.is_alpha and (
                is_known_abbreviation
                or (is_short_word and not is_common_german_word)
            )

            if is_plausible_stem:
                # Skip if inside quotation marks
                if token.i < len(doc) - 1:
                    next_token = doc[token.i + 1]
                    if next_token.text in ALLOWED_QUOTATION_MARKS:
                        continue

                abbreviation = f"{prev_token.text}{token.text}"

                if abbreviation not in found:
                    errors.append(
                        f'Abkürzung "{abbreviation}" gefunden. '
                        f'Besser: Schreiben Sie "{prev_token.text}" vollständig aus. '
                        f"Leichte Sprache vermeidet Abkürzungen."
                    )
                    found.add(abbreviation)

        # Method 2: Token contains period (spaCy tokenization like "Dr.", "z.B.")
        elif "." in token.text and token.text not in [".", "...", "…"]:
            # Check if it is really an abbreviation (not URL, number with decimal)
            if (
                token.text.endswith(".")
                and len(token.text) > 1
                and not token.text[-2].isdigit()
            ):
                # Single uppercase letters at sentence end are NOT abbreviations
                stem = token.text[:-1]
                if len(stem) == 1 and token.is_sent_end:
                    continue

                abbreviation = token.text

                if abbreviation not in found:
                    errors.append(
                        f'Abkürzung "{abbreviation}" gefunden. '
                        f'Besser: Schreiben Sie "{stem}" vollständig aus. '
                        f"Leichte Sprache vermeidet Abkürzungen."
                    )
                    found.add(abbreviation)

    return errors


# =========================================================================
# MAIN FUNCTION: WHITELIST-BASIERTE ERKENNUNG
# =========================================================================


def check_rule(doc: Doc) -> List[str]:
    """
    WHITELIST-BASIERTE Erkennung von Interpunktions-Problemen (v3.0).

    FUNDAMENTAL-ÄNDERUNG:
    - Prüft JEDES Zeichen im Text
    - NUR explizit erlaubte Zeichen (., ?, !, :, Anführungszeichen, ·) werden akzeptiert
    - ALLE anderen Zeichen werden als Fehler gemeldet
    - Extrem gründlich: Erkennt JEDE Art von Sonderzeichen, Symbol, Unicode-Zeichen

    LEICHTE SPRACHE REGEL:
    "Zur Verfügung stehen Punkt, Frage-, Ausrufezeichen, Doppelpunkt, Anführungszeichen, Mediopunkt."

    Args:
        doc: spaCy Doc-Objekt

    Returns:
        Liste von Fehlermeldungen
    """
    errors = []
    processed_chars = {}

    # -----------------------------------------------------------------------
    # 1. DETECT ABBREVIATIONS (always forbidden)
    # -----------------------------------------------------------------------
    abbreviation_errors = detect_abbreviations(doc)
    errors.extend(abbreviation_errors)

    # -----------------------------------------------------------------------
    # 2. COMMA STRUCTURE ANALYSIS (max. 2 commas per sentence)
    # -----------------------------------------------------------------------
    comma_issues = analyze_comma_structures(doc)
    errors.extend(comma_issues)

    # -----------------------------------------------------------------------
    # 3. WHITELIST CHECK: CHECK EVERY CHARACTER IN TEXT
    # -----------------------------------------------------------------------

    # SPECIAL CASE: Time expressions (14:30, 09:15, etc.) - allow colon in numbers
    # Find all time expressions in text (HH:MM format)
    time_positions = set()
    for match in re.finditer(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", doc.text):
        # Mark all character positions in the time expression
        for pos in range(match.start(), match.end()):
            time_positions.add(pos)

    # Iterate over EVERY character in the entire text (not just tokens!)
    for i, char in enumerate(doc.text):
        # IGNORE: Characters in time expressions
        if i in time_positions:
            continue

        # IGNORE: Letters, digits, whitespace
        if is_letter_or_digit(char) or char in TECHNICAL_CHARACTERS:
            continue

        # ALLOWED CHARACTERS: Period, ?, !, :, mediopunkt, quotation marks (typographic)
        if char in ALLOWED_PUNCTUATION_MARKS or char in ALLOWED_QUOTATION_MARKS:
            continue

        # CONTEXT-DEPENDENT: Comma
        if char in CONTEXT_ALLOWED_CHARACTERS:
            # COMMA: Always skip (checked in analyze_comma_structures())
            if char == ",":
                continue

        # CONTEXT-DEPENDENT: Hyphen in compounds
        # Allow hyphens within compound words (surrounded by letters)
        if char == "-":
            # Check if hyphen is between letters (compound word)
            if i > 0 and i < len(doc.text) - 1:
                prev_char = doc.text[i - 1]
                next_char = doc.text[i + 1]
                if prev_char.isalpha() and next_char.isalpha():
                    continue  # Skip - this is a compound word hyphen

        # -----------------------------------------------------------------------
        # FORBIDDEN CHARACTER FOUND!
        # -----------------------------------------------------------------------
        char_info = classify_forbidden_character(char)

        # Count occurrences per character type
        if char not in processed_chars:
            processed_chars[char] = 0
        processed_chars[char] += 1

    # Generate error messages with count
    for char, count in processed_chars.items():
        char_info = classify_forbidden_character(char)
        if count > 1:
            errors.append(
                f'{char_info["bezeichnung"]} "{char_info["zeichen"]}" gefunden ({count} Mal). '
                f"Besser: {char_info['vorschlag']} "
                f"(Unicode: {char_info['unicode_name']})"
            )
        else:
            errors.append(
                f'{char_info["bezeichnung"]} "{char_info["zeichen"]}" gefunden. '
                f"Besser: {char_info['vorschlag']} "
                f"(Unicode: {char_info['unicode_name']})"
            )

    # -----------------------------------------------------------------------
    # 4. PATTERN-BASED DETECTION: Multiple punctuation marks
    # -----------------------------------------------------------------------
    text = doc.text

    # Multiple periods, question marks, exclamation marks
    repeated_patterns = re.findall(r"[.!?]{2,}", text)
    for pattern in set(repeated_patterns):
        if pattern not in ["..."]:  # ... is reported separately as ellipsis
            errors.append(
                f'Mehrfache Satzzeichen "{pattern}" sind verwirrend. '
                f"Besser: Ein Satzzeichen genügt. "
                f"Leichte Sprache verwendet einfache Satzzeichen."
            )

    # Ellipsis (...)
    if "..." in text or "…" in text:
        errors.append(
            'Auslassungspunkte "..." oder "…" gefunden. '
            "Besser: Schreiben Sie Gedanken vollständig aus. "
            "Leichte Sprache vermeidet Auslassungen."
        )

    # Emails and URLs (often contain @, which are reported separately)
    for token in doc:
        if token.like_email:
            errors.append(
                f'E-Mail-Adresse "{token.text}" im Fließtext gefunden. '
                f'Besser: Schreiben Sie "Schreiben Sie uns eine E-Mail" oder geben Sie die Adresse separat an.'
            )
        elif token.like_url:
            errors.append(
                f'URL "{token.text}" im Fließtext gefunden. '
                f'Besser: Geben Sie die Adresse separat an oder schreiben Sie "Besuchen Sie unsere Webseite".'
            )

    return errors


# =========================================================================
# TESTING
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import spacy

    logger.info("=" * 80)
    logger.info("INTERPUNKTIONS-ERKENNUNG v3.0 - WHITELIST-BASIERT")
    logger.info("=" * 80)
    logger.info("ERLAUBT: . ? ! : Anführungszeichen(typografisch) ·")
    logger.info('VERBOTEN: Gerade Anführungszeichen " und alle anderen Sonderzeichen')
    logger.info("=" * 80)

    try:
        nlp = spacy.load("de_core_news_lg")

        # Umfassende Test-Texte
        test_texte = [
            # ERLAUBTE Zeichen (sollten OK sein)
            ("Das ist ein guter Satz.", "Punkt am Ende (OK)"),
            ("Ist das richtig?", "Fragezeichen (OK)"),
            ("Das ist wichtig!", "Ausrufezeichen (OK)"),
            ('Er sagte: "Das ist gut."', "Anführungszeichen (OK)"),
            ("Die Punkte sind: Punkt 1, Punkt 2.", "Doppelpunkt mit Liste (OK)"),
            ("Das Meeting ist um 14:30 Uhr.", "Doppelpunkt für Zeit (OK)"),
            # VERBOTENE Zeichen (sollten Fehler sein)
            ("Das ist wichtig; aber schwer.", "Semikolon (Fehler)"),
            ("Das ist wichtig (siehe Anhang).", "Klammern (Fehler)"),
            ("Der Preis beträgt 50€.", "Euro-Zeichen (Fehler)"),
            ("Kontakt: info@firma.de", "E-Mail (Fehler)"),
            ("Mann/Frau bitte ausfüllen.", "Schrägstrich (Fehler)"),
            ("Mitarbeiter*innen sind willkommen.", "Genderstern (Fehler)"),
            ("Das kostet 50$ pro Stück.", "Dollar-Zeichen (Fehler)"),
            ("Siehe § 5 des Gesetzes.", "Paragraph-Zeichen (Fehler)"),
            ("Das ist © geschützt.", "Copyright (Fehler)"),
            ("Der Wert ist > 100.", "Größer-als (Fehler)"),
            ("Die Temperatur ist 20°C.", "Grad-Zeichen (Fehler)"),
            ("Das ergibt 5 + 3 = 8.", "Plus und Gleichheitszeichen (Fehler)"),
            ("Das bedeutet → mehr Erfolg.", "Pfeil (Fehler)"),
            ("Bewertung: ★★★★★", "Sterne (Fehler)"),
            ("50% der Menschen.", "Prozent-Zeichen (Fehler)"),
            ("Firma & Co. KG", "Ampersand (Fehler)"),
            ("Das ist #wichtig.", "Hashtag (Fehler)"),
            ("Die Firma – ein großes Unternehmen – plant.", "Gedankenstrich (Fehler)"),
            # Mehrfache Kommas
            (
                "Das ist gut, aber schwer, und lang, mit vielen Kommas.",
                "Zu viele Kommas (Fehler)",
            ),
            # Abkürzungen
            ("Dr. Müller kommt morgen.", "Abkürzung (Fehler)"),
            ("Das ist z.B. wichtig.", "Abkürzung (Fehler)"),
        ]

        logger.info("\n--- INTERPUNKTIONS-TESTS ---\n")

        for i, (text, beschreibung) in enumerate(test_texte, 1):
            logger.info("[%s/%s] %s", i, len(test_texte), text)
            logger.info(" Test: %s", beschreibung)

            doc = nlp(text)
            results = check_rule(doc)

            if results:
                for error in results:
                    logger.error(" %s", error)
            else:
                logger.info(" Kein Problem gefunden (akzeptabel)")
            logger.info("")

        logger.info("=" * 80)
        logger.info("TESTS ABGESCHLOSSEN")
        logger.info("=" * 80)

    except OSError:
        logger.error(" spaCy-Modell 'de_core_news_lg' nicht gefunden!")
        logger.error(" Bitte ausführen: python -m spacy download de_core_news_lg")
