"""
Rule for detecting multiple statements in a single sentence.

According to Leichte Sprache rules, a sentence should ideally contain only one statement.
This rule uses a trained BERT model based on the StaGE dataset
(Statement Segmentation in German Easy Language).

Source: https://german-easy-to-read.github.io/statements/
Model training: stage_statement_model/
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
import torch
import torch.nn as nn
from spacy.tokens import Doc, Span
from transformers import AutoModel, AutoTokenizer

from regeln.base_ml_rule import BaseMLRule

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ============================================================================
# CONFIGURATION
# ============================================================================
# Adjust these values to change the behavior of the rule

# Should the full sentence be shown in the output?
# True  = Shows the entire sentence (regardless of length)
# False = Shows only the first 80 characters + "..." (default)
SHOW_FULL_SENTENCE = True

# Semantic word fields for coherence checking
# Verbs from the same word field = semantically related = often one statement
SEMANTIC_WORD_FIELDS = {
    # Allgemeine Wortfelder
    "überwachung": {
        "beobachten",
        "kontrollieren",
        "prüfen",
        "überwachen",
        "checken",
        "überprüfen",
        "beaufsichtigen",
    },
    "teilhabe": {
        "dabei",
        "teilnehmen",
        "mitmachen",
        "beteiligen",
        "mitwirken",
        "teilhaben",
        "sein",
        "dabeisein",
    },
    "grundfertigkeiten": {
        "lesen",
        "schreiben",
        "rechnen",
        "sprechen",
        "hören",
        "zuhören",
    },
    "kommunikation": {
        "sprechen",
        "reden",
        "erzählen",
        "berichten",
        "mitteilen",
        "sagen",
        "informieren",
        "erklären",
    },
    "freude": {"spielen", "lachen", "freuen", "spaß", "vergnügen", "jubeln", "feiern"},
    "bewegung": {"gehen", "laufen", "rennen", "springen", "hüpfen", "bewegen"},
    "wahrnehmung": {
        "sehen",
        "hören",
        "fühlen",
        "spüren",
        "riechen",
        "schmecken",
        "merken",
        "bemerken",
    },
    "lernen": {"lernen", "verstehen", "begreifen", "üben", "studieren", "trainieren"},
    # Leichte-Sprache-spezifische Redundanz-Gruppen
    "verstehen_gruppe": {
        "verstehen",
        "begreifen",
        "kapieren",
        "wissen",
        "kennen",
        "erkennen",
    },
    "helfen_gruppe": {
        "helfen",
        "unterstützen",
        "beistehen",
        "assistieren",
        "fördern",
        "ermöglichen",
        "geben",
        "machen",
        "vorschlagen",
        "tipp",
    },
    "wichtig_gruppe": {
        "wichtig",
        "bedeutsam",
        "bedeutend",
        "wesentlich",
        "relevant",
        "nötig",
    },
    "organisieren_gruppe": {
        "organisieren",
        "planen",
        "vorbereiten",
        "regeln",
        "arrangieren",
        "koordinieren",
    },
    "entscheiden_gruppe": {
        "entscheiden",
        "bestimmen",
        "wählen",
        "auswählen",
        "festlegen",
    },
    "arbeiten_gruppe": {"arbeiten", "tun", "machen", "erledigen", "schaffen", "wirken"},
    "zeigen_gruppe": {
        "zeigen",
        "vorzeigen",
        "präsentieren",
        "darstellen",
        "demonstrieren",
    },
    "beginnen_gruppe": {"beginnen", "anfangen", "starten", "loslegen", "einleiten"},
}

# Erweiterte Wortfelder speziell für Leichte Sprache (häufige Redundanzen)
EASY_LANGUAGE_SPECIFIC = {
    "haben_können": {
        "haben",
        "können",
        "dürfen",
    },  # "Menschen haben Rechte und können wählen"
    "sein_werden": {"sein", "werden", "bleiben"},  # "ist wichtig und wird beachtet"
    "müssen_sollen": {
        "müssen",
        "sollen",
        "brauchen",
    },  # "muss geprüft und soll verbessert werden"
}

# ============================================================================


class BERTStatementClassifier(nn.Module):
    """BERT-basierter Classifier für Statement-Segmentation."""

    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def _get_model_path() -> Optional[Path]:
    """Findet den Pfad zum trainierten Modell."""
    # Standalone-fähig: Modell liegt im selben Ordner wie regel.py
    model_dir = (
        Path(__file__).parent / "model" / "models" / "stage_statement_classifier_best"
    )

    if model_dir.exists():
        return model_dir

    # Fallback: Suche direkt im model-Ordner (falls Struktur anders ist)
    fallback_dir = Path(__file__).parent / "model" / "stage_statement_classifier_best"
    if fallback_dir.exists():
        return fallback_dir

    return None


class MultipleStatementsModel(BaseMLRule):
    """Thread-safe lazy loading for the StaGE statement classifier."""

    @classmethod
    def _load_model(cls):
        """Load the trained BERT model for statement segmentation.

        Returns:
            Tuple of (model, tokenizer, device)

        Raises:
            FileNotFoundError: If the StaGE model files are not found.
        """
        model_path = _get_model_path()

        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                "Trainiertes StaGE-Modell nicht gefunden. "
                "Bitte trainieren Sie das Modell mit: "
                "stage_statement_model/scripts/train_model_simple.py"
            )

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lade Config
        with open(model_path / "config.json", "r") as f:
            model_config = json.load(f)

        # Lade Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Lade Model
        model = BERTStatementClassifier(
            model_name=model_config["model_name"],
            num_labels=model_config["num_labels"],
        )

        # Lade Weights
        checkpoint = torch.load(
            model_path / "pytorch_model.bin", map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info("StaGE-Modell geladen von: %s", model_path)

        return (model, tokenizer, device)


def _detect_text_type(doc: Doc) -> str:
    """
    Erkennt, ob es sich um normale Sprache oder Leichte Sprache handelt.

    Args:
        doc: spaCy Doc-Objekt mit dem gesamten Text

    Returns:
        'leichte_sprache' oder 'normale_sprache'
    """
    # Sammle Indikatoren
    indicators = {"leichte_sprache": 0, "normale_sprache": 0}

    # 1. Durchschnittliche Satzlänge
    sents = list(doc.sents)
    if sents:
        avg_sent_length = sum(
            len([t for t in s if not t.is_punct]) for s in sents
        ) / len(sents)
        if avg_sent_length < 10:
            indicators["leichte_sprache"] += 2
        elif avg_sent_length > 15:
            indicators["normale_sprache"] += 2

    # 2. Middot-Zeichen (·) für Worttrennung
    if "·" in doc.text or "∙" in doc.text:
        indicators["leichte_sprache"] += 3  # Starker Indikator

    # 3. Komplexe Nebensätze (weil, damit, obwohl, etc.)
    nebensatz_marker = {
        "weil",
        "damit",
        "obwohl",
        "nachdem",
        "bevor",
        "während",
        "falls",
        "sofern",
    }
    for token in doc:
        if token.text.lower() in nebensatz_marker:
            indicators["normale_sprache"] += 1

    # 4. Doppelpunkte als Einleitungen
    doppelpunkt_count = doc.text.count(":")
    if doppelpunkt_count > len(sents) / 10:  # Mehr als 1 Doppelpunkt pro 10 Sätze
        indicators["leichte_sprache"] += 1

    # 5. Aufzählungszeichen
    if any(marker in doc.text for marker in ["•", "○", "-", "►", "▪"]):
        indicators["leichte_sprache"] += 1

    # 6. Einfache Satzstrukturen (nur Hauptsätze)
    has_subordinate = False
    for sent in sents[: min(10, len(sents))]:  # Prüfe erste 10 Sätze
        for token in sent:
            if token.dep_ in ["cp", "cj"] and token.pos_ == "SCONJ":
                has_subordinate = True
                break

    if not has_subordinate and len(sents) > 5:
        indicators["leichte_sprache"] += 1

    # Entscheidung
    if indicators["leichte_sprache"] > indicators["normale_sprache"]:
        return "leichte_sprache"
    else:
        return "normale_sprache"


def _preprocess_leichte_sprache(text: str) -> str:
    """
    Preprocessing für Leichte-Sprache-spezifische Formatierungen.

    Entfernt oder ersetzt Sonderzeichen, die das BERT-Modell nicht kennt:
    - Middot (·) für Worttrennung → wird entfernt
    - Andere Leichte-Sprache-spezifische Zeichen

    Args:
        text: Original-Text

    Returns:
        Bereinigter Text für BERT-Verarbeitung
    """
    # Entferne Middot (·) - wird als [UNK] Token behandelt
    text = text.replace("·", "")

    # Normalisiere Whitespace
    text = " ".join(text.split())

    return text


def _calculate_adaptive_threshold(
    sent: Span, base_confidence: float, doc: Doc = None
) -> float:
    """
    Berechnet einen adaptiven Threshold basierend auf linguistischen Features und Text-Typ.

    ÄNDERUNG (2025-10-21): Berücksichtigt Text-Typ (normale vs. Leichte Sprache)
    - Semantische Kohärenz
    - Mehrere Subjekte
    - Kontrastive Konjunktionen
    - Kopula-Konstruktionen
    - Text-Typ-basierte Anpassung

    Args:
        sent: spaCy Span-Objekt (Satz)
        base_confidence: Die BERT-Konfidenz
        doc: Optionales Doc-Objekt für Text-Typ-Erkennung

    Returns:
        Angepasster Threshold (0.4-0.95)
    """
    # Text-Typ erkennen (wenn Doc verfügbar)
    text_type = "normale_sprache"  # Default
    text_type_adjustment = 0.0

    if doc is not None:
        text_type = _detect_text_type(doc)
        if text_type == "leichte_sprache":
            # Bei Leichter Sprache: Leicht konservativere Thresholds
            text_type_adjustment = 0.05  # Moderate Erhöhung für Leichte Sprache

    # Extrahiere linguistische Features
    features = _analyze_linguistic_features(sent)

    # PRINZIP 1: Strukturelle Multi-Statement Indikatoren (sehr sensitiv)
    if features["num_subjects"] > 1 or features["has_contrast"]:
        # Mehrere Subjekte oder Kontrast = sehr wahrscheinlich Multi-Statement
        # Text-Typ hat wenig Einfluss bei klaren strukturellen Indikatoren
        return 0.50 + (text_type_adjustment * 0.5)  # Leicht angepasst für Text-Typ

    # PRINZIP 2: Semantische Kohärenz (moderat konservativ)
    if features["semantically_coherent"]:
        # Verben aus demselben Wortfeld = wahrscheinlich ein Statement
        # Aber nicht zu aggressiv für normale Sprache
        base_threshold = 0.80 + text_type_adjustment  # Bei Leichter Sprache: 0.85

        # Zusätzliche Anpassung basierend auf Verb-Distanz
        if features["verbs_close_together"]:
            # Verben direkt nacheinander = noch wahrscheinlicher ein Statement
            base_threshold += 0.05
        elif features["verbs_far_apart"]:
            # Verben weit auseinander = etwas weniger sicher
            base_threshold -= 0.05

        return min(base_threshold, 1.0)  # Cap bei 1.0

    # PRINZIP 3: Syntaktische Einfachheit
    if features["is_copula"]:
        # Kopula-Konstruktion = Eine Eigenschaftsaussage
        # In Leichter Sprache sehr häufig
        return 0.95 + (text_type_adjustment * 0.5)  # Bei Leichter Sprache: 1.0

    # Standard-Fallbacks (aus alter Implementierung)
    text_lower = features["text_lower"]

    # Spezielle Patterns in Leichter Sprache
    # "alle Menschen mit Behinderungen" ist ein häufiges Pattern
    if "alle menschen" in text_lower or "allen menschen" in text_lower:
        # Sehr konservativ bei diesem Pattern
        return 0.95 + text_type_adjustment

    # Modalverben ohne Konjunktion
    if features["has_modalverb"] and not features["has_coord_conjunction"]:
        if text_lower.count(" alle ") >= 2:
            return 0.90 + text_type_adjustment
        return 0.85 + text_type_adjustment

    # Bei Koordination ohne semantische Kohärenz
    if features["has_coord_conjunction"] and features["num_commas"] >= 2:
        return 0.70 + text_type_adjustment  # Bei Leichter Sprache: 0.80

    # Standard
    return 0.75 + text_type_adjustment  # Bei Leichter Sprache: 0.85


def _predict_multiple_statements(
    sent: Span, doc: Doc = None, threshold: float = 0.7
) -> tuple[bool, float]:
    """
    Sagt vorher, ob ein Satz mehrere Statements enthält.

    ÄNDERUNGEN (2025-10-21):
    - Nimmt jetzt Span statt Text
    - Nutzt adaptive Thresholds basierend auf linguistischen Features
    - Berücksichtigt Text-Typ (normale vs. Leichte Sprache)

    Args:
        sent: spaCy Span-Objekt (Satz)
        doc: Optionales Doc-Objekt für Text-Typ-Erkennung
        threshold: Basis-Konfidenz-Schwellenwert (Standard: 0.7, wird adaptiv angepasst)

    Returns:
        (has_multiple_statements, confidence)
    """
    text = sent.text.strip()
    result = MultipleStatementsModel.get_model()

    if result is None:
        # Fallback: Heuristische Methode
        return _heuristic_multiple_statements(text), 0.5

    model, tokenizer, device = result

    if model is None or tokenizer is None:
        # Fallback: Heuristische Methode
        return _heuristic_multiple_statements(text), 0.5

    try:
        # Preprocessing für Leichte Sprache
        text_clean = _preprocess_leichte_sprache(text)

        # Tokenisiere
        encoding = tokenizer(
            text_clean,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Vorhersage
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][
                1
            ].item()  # Wahrscheinlichkeit für "mehrere Statements"

        # Adaptive Threshold-Berechnung basierend auf linguistischen Features und Text-Typ
        adaptive_threshold = _calculate_adaptive_threshold(sent, confidence, doc)
        has_multiple = confidence >= adaptive_threshold

        return has_multiple, confidence

    except Exception as e:
        logger.warning("Fehler bei Vorhersage: %s", e)
        return _heuristic_multiple_statements(text), 0.5


def _check_semantic_coherence(
    verb_lemmas: List[str],
) -> Tuple[bool, Optional[str], Optional[set]]:
    """
    Prüft, ob koordinierte Verben semantisch kohärent sind (aus demselben Wortfeld).

    Args:
        verb_lemmas: Liste der Verb-Lemmas

    Returns:
        (is_coherent, field_name, overlapping_verbs):
            - is_coherent: True wenn Verben aus demselben Wortfeld
            - field_name: Name des semantischen Wortfelds
            - overlapping_verbs: Die gefundenen verwandten Verben
    """
    verb_set = set(v.lower() for v in verb_lemmas)

    for field_name, field_words in SEMANTIC_WORD_FIELDS.items():
        overlap = verb_set & field_words
        if len(overlap) >= 2:
            return True, field_name, overlap

    return False, None, None


def _heuristic_multiple_statements(text: str) -> bool:
    """
    Verbesserte heuristische Fallback-Methode zur Erkennung mehrerer Statements.

    ÄNDERUNGEN (2025-10-13):
    - Konservativer: Nur bei expliziten Konjunktionen + Länge
    - Vermeidet False Positives bei Modalverb-Sätzen

    Basiert auf:
    - Explizite Konjunktionen ("und", "aber", "oder") + Satzlänge
    - Mehrere Kommata (mindestens 2)
    - Konservativ: Modalverben allein lösen nicht aus
    """
    text_lower = text.lower()
    wort_anzahl = len(text.split())

    # Starke Indikatoren (explizite Konjunktionen)
    strong_conjunctions = [" und ", " aber ", " oder ", " doch "]
    has_strong_conjunction = any(conj in text_lower for conj in strong_conjunctions)

    # Nur bei starken Indikatoren UND langem Satz
    if has_strong_conjunction and wort_anzahl > 10:
        return True

    # Mehrere Kommata (mindestens 2) - sehr starker Indikator
    if text.count(",") >= 2:
        return True

    # Ansonsten konservativ: Ein Statement
    # Vermeidet False Positives bei:
    # - Modalverben ohne Konjunktionen
    # - Präpositionalphrasen ("für das", "in einem")
    # - Attributen ("mit Behinderungen")
    return False


def _analyze_linguistic_features(sent: Span) -> Dict:
    """
    Extrahiert linguistische Features aus einem spaCy Sentence Span.

    Nutzt spaCy's Dependency Parsing und POS-Tagging für präzise Analyse:
    - Koordinierte Konjunktionen (CCONJ)
    - Koordinierte Verben (dep='cj')
    - Expletives "es"
    - Modalverben
    - Satzstruktur

    Args:
        sent: spaCy Span-Objekt (Satz)

    Returns:
        Dict mit linguistischen Features
    """
    features = {
        "text": sent.text,
        "text_lower": sent.text.lower(),
        "length": len(sent),
        "num_tokens": len(
            [t for t in sent if not t.is_punct and not t.is_space]
        ),  # Exclude punctuation AND whitespace
        "num_commas": sent.text.count(","),
        "sent": sent,  # Speichere das Span-Objekt selbst für Debugging
    }

    # Extrahiere Tokens
    tokens = [t for t in sent]

    # Koordinierte Konjunktionen (und, aber, oder, doch)
    # WICHTIG: Sentence-initial CCONJs (z.B. "Und" am Satzanfang) zählen NICHT
    # als interne Koordination - sie verbinden Sätze, nicht Statements innerhalb eines Satzes
    coord_conjunctions = [t for t in tokens if t.pos_ == "CCONJ"]

    # Filtere sentence-initial CCONJ aus (erstes Token im Satz)
    internal_coord_conjunctions = [
        t
        for t in coord_conjunctions
        if t.i != sent[0].i  # Nicht das erste Token
    ]

    features["has_coord_conjunction"] = len(internal_coord_conjunctions) > 0
    features["coord_conjunctions"] = [t.text for t in internal_coord_conjunctions]
    features["has_sentence_initial_cconj"] = len(coord_conjunctions) > len(
        internal_coord_conjunctions
    )

    # Verben und Auxiliare
    verbs = [t for t in tokens if t.pos_ in ["VERB", "AUX"]]
    features["num_verbs"] = len(verbs)
    features["verbs"] = [t.text for t in verbs]
    features["verb_lemmas"] = [t.lemma_ for t in verbs]  # Lemmas für Modalverb-Check

    # Koordinierte Verben (mehrere Prädikate)
    coordinated_verbs = [t for t in verbs if t.dep_ == "cj"]
    features["has_coordinated_verbs"] = len(coordinated_verbs) > 0
    features["coordinated_verbs"] = [t.text for t in coordinated_verbs]

    # Verb-Distanz-Analyse (für koordinierte Verben)
    features["verb_distance"] = None
    if len(verbs) >= 2 and features["has_coordinated_verbs"]:
        # Berechne minimale Distanz zwischen koordinierten Verben
        verb_positions = [v.i for v in verbs]
        if len(verb_positions) >= 2:
            min_distance = min(
                verb_positions[i + 1] - verb_positions[i]
                for i in range(len(verb_positions) - 1)
            )
            features["verb_distance"] = min_distance
            # < 3 Tokens: direkt nacheinander (z.B. "beobachten und kontrollieren")
            # >= 5 Tokens: weit auseinander (z.B. "arbeitet im Büro und geht nach Hause")
            features["verbs_close_together"] = min_distance < 3
            features["verbs_far_apart"] = min_distance >= 5
    else:
        features["verbs_close_together"] = False
        features["verbs_far_apart"] = False

    # Expletives "es" (muss es, gibt es, etc.)
    expletive_es = any(t.text.lower() == "es" and t.pos_ == "PRON" for t in tokens)
    features["has_expletive_es"] = expletive_es

    # Modalverben (inkl. "möchten" - häufig in Leichter Sprache)
    # WICHTIG: Nutze Lemmas (möchte → möchten), nicht Text!
    modalverbs = [
        "können",
        "müssen",
        "sollen",
        "dürfen",
        "wollen",
        "mögen",
        "möchten",
        "muss",
    ]
    features["has_modalverb"] = any(
        lemma.lower() in modalverbs for lemma in features["verb_lemmas"]
    )

    # Modalverb-Infinitiv-Konstruktionen (muss abbauen, kann lesen)
    # Diese zählen als EINE Verbphrase, nicht zwei separate Aktionen
    features["is_modal_infinitive"] = False
    features["effective_verb_count"] = len(verbs)  # Default: alle Verben zählen

    if features["has_modalverb"]:
        # Check für Infinitive (dep='oc' = objektiver Infinitiv)
        infinitives = [t for t in tokens if t.dep_ == "oc" and t.pos_ == "VERB"]
        # Modalverb + Infinitiv(e) ohne koordinierte Konjunktion = eine Verbphrase
        # z.B. "muss abbauen", "kann verstehen"
        # ABER: "kann lesen und schreiben" mit semantischer Kohärenz = auch OK
        if infinitives and len(infinitives) == 1:
            features["is_modal_infinitive"] = True
            # Modalverb + einzelner Infinitiv = 1 effektive Aktion
            features["effective_verb_count"] = 1
            features["modal_infinitive_verbs"] = [t.text for t in infinitives]

    # Perfekt/Auxiliar-Konstruktionen (hat gemacht, wurde gesehen)
    # WICHTIG: NUR wenn es wirklich nur EINE Aktion ist
    # "hat gemacht" = Perfekt (eine Zeitform)
    # "hat entdeckt und gemeldet" = zwei Aktionen (trotz Perfekt)
    features["is_perfect_tense"] = False
    aux_verbs = [
        t
        for t in verbs
        if t.pos_ == "AUX" and t.lemma_.lower() in ["haben", "werden", "sein"]
    ]

    # Check für Partizip II (gemacht, gesehen, entdeckt)
    participles = []
    for token in tokens:
        # Partizip II erkennen: endet oft auf -t oder -en nach Auxiliar
        if token.pos_ == "VERB" and token.dep_ in ["oc", "ROOT"]:
            if any(aux.i < token.i for aux in aux_verbs):  # Nach einem Auxiliar
                participles.append(token)

    # NUR als Perfekt markieren wenn:
    # 1. Es gibt ein Auxiliar (hat/wurde/ist)
    # 2. Es gibt genau EIN Partizip (nicht mehrere koordinierte)
    # 3. Keine koordinierten Verben
    if aux_verbs and len(participles) == 1 and not features["has_coordinated_verbs"]:
        features["is_perfect_tense"] = True
        # Auxiliar + einzelnes Partizip = 1 effektive Aktion
        if not features["is_modal_infinitive"]:  # Nicht doppelt reduzieren
            features["effective_verb_count"] = 1

    # "auch" als Adverb (Fokuspartikel)
    features["has_auch"] = any(
        t.text.lower() == "auch" and t.pos_ == "ADV" for t in tokens
    )

    # Root des Satzes
    root_tokens = [t for t in tokens if t.dep_ == "ROOT"]
    features["root"] = root_tokens[0].text if root_tokens else None
    features["root_pos"] = root_tokens[0].pos_ if root_tokens else None

    # Subjekte (mehrere verschiedene Subjekte = starker Multi-Statement Indikator)
    subjects = [t for t in tokens if t.dep_ == "sb"]
    features["num_subjects"] = len(subjects)
    features["subjects"] = [t.text for t in subjects]

    # Kopula-Konstruktionen (sein + Adjektiv/Prädikativ)
    # z.B. "ist wichtig", "sind gut" = Eine Eigenschaftsaussage
    is_copula = False
    is_copula_predicate = False

    # WICHTIG: Prüfe zuerst auf Nebensätze - keine Korrektur bei subordinate clauses!
    has_subordinate_clause = any(t.pos_ == "SCONJ" for t in tokens)

    if (
        features["root_pos"] == "AUX"
        and features["root"]
        and features["root"].lower() in ["sein", "ist", "sind", "war", "waren"]
    ):
        # Check für Prädikativ (pd) oder Adjektiv
        has_predicate = any(t.dep_ == "pd" for t in tokens)
        if has_predicate:
            is_copula = True

        # WICHTIG: Bei "Das/Es sind X" wird X oft fälschlich als Subjekt erkannt
        # Korrektur: Wenn Kopula und zweites "Subjekt" nach Kopula → ist Prädikativ
        # ABER: NUR wenn kein Nebensatz vorliegt!

        # WICHTIG: Verwende relative Positionen innerhalb des Spans!
        # sent.start gibt den Offset im Original-Doc an
        sent_start = sent.start if hasattr(sent, "start") else 0
        root_position_relative = root_tokens[0].i - sent_start if root_tokens else -1

        false_subjects = []
        for subj in subjects:
            # Verwende relative Position innerhalb des Spans
            subj_position_relative = subj.i - sent_start

            # Wenn "Subjekt" nach Kopula-Verb kommt → wahrscheinlich Prädikativ
            if subj_position_relative > root_position_relative and subj.pos_ == "NOUN":
                false_subjects.append(subj)
                is_copula_predicate = True

        # Korrigiere Subjekt-Anzahl - ABER NUR wenn kein Nebensatz vorliegt!
        if is_copula_predicate and false_subjects and not has_subordinate_clause:
            # WICHTIG: Korrigiere die Anzahl der Subjekte
            corrected_count = len(subjects) - len(false_subjects)
            features["num_subjects"] = max(
                1, corrected_count
            )  # Mindestens 1 Subjekt bleibt
            features["false_subjects_as_predicate"] = [t.text for t in false_subjects]
        elif is_copula_predicate and false_subjects and has_subordinate_clause:
            # Bei Nebensätzen: Keine Korrektur, da mehrere Subjekte legitim sind
            is_copula_predicate = False  # Deaktiviere die falsche Markierung

    features["is_copula"] = is_copula
    features["is_copula_predicate"] = is_copula_predicate

    # Kontrastive Konjunktionen (starker Multi-Statement Indikator)
    contrast_words = {"aber", "jedoch", "sondern", "während", "dagegen", "hingegen"}
    features["has_contrast"] = any(t.text.lower() in contrast_words for t in tokens)

    # Vergleichskonstruktionen (wie, als) = meist ein Statement
    # z.B. "die gleichen Rechte wie andere Menschen"
    features["has_comparison"] = any(
        t.text.lower() in ["wie", "als"] and t.pos_ in ["CCONJ", "SCONJ", "ADV", "ADP"]
        for t in tokens
    )

    # Semantische Kohärenz der Verben prüfen
    if features["has_coordinated_verbs"]:
        is_coherent, field_name, overlap = _check_semantic_coherence(
            features["verb_lemmas"]
        )
        features["semantically_coherent"] = is_coherent
        features["semantic_field"] = field_name
        features["coherent_verbs"] = overlap
    else:
        features["semantically_coherent"] = False
        features["semantic_field"] = None
        features["coherent_verbs"] = None

    return features


def _is_obvious_single_statement(sent: Span) -> Tuple[bool, Optional[str]]:
    """
    Pre-Filter: Identifiziert Sätze, die OFFENSICHTLICH nur ein Statement haben.

    Nutzt spaCy-basierte linguistische Analyse + String-Patterns für präzise
    Erkennung von typischen Ein-Statement-Strukturen in Leichter Sprache.

    ÄNDERUNGEN (2025-10-13 - Option 3 Feature Gating):
    - Nutzt spaCy Dependency Parsing statt nur String-Matching
    - Berücksichtigt syntaktische Struktur (koordinierte Verben, etc.)
    - Präziser als reine String-basierte Heuristik

    Args:
        sent: spaCy Span-Objekt (Satz)

    Returns:
        (is_obvious, reason):
            - True = Definitiv ein Statement (Skip BERT)
            - False = Unsicher (BERT fragen)
            - reason = Erklärung für Logging
    """
    # Extrahiere linguistische Features
    features = _analyze_linguistic_features(sent)

    text_lower = features["text_lower"]
    num_tokens = features["num_tokens"]
    num_commas = features["num_commas"]

    # ======================================================================
    # RULE -1: Sentences with subordinate clauses (highest priority)
    # ======================================================================
    # If a sentence has a subordinate clause (SCONJ), the nebensaetze rule
    # will already flag it. No need for mehrere_aussagen to double-flag.
    has_subordinate_clause = any(t.pos_ == "SCONJ" for t in sent)
    if has_subordinate_clause:
        return True, "Has subordinate clause (handled by nebensaetze rule)"

    # ======================================================================
    # RULE 0: Spezielle grammatische Konstruktionen (höchste Priorität)
    # ======================================================================
    # Diese Konstruktionen sind IMMER ein Statement, unabhängig von anderen Features
    # NOTE: has_subordinate_clause is always False here (we returned early in RULE -1 if True)

    # Kopula-Prädikativ-Konstruktion (Das sind die Rechte)
    if features.get("is_copula_predicate", False):
        return (
            True,
            "Kopula-Prädikativ-Konstruktion (falsche Subjekt-Erkennung korrigiert)",
        )

    # Zusätzlicher Check: "Das sind..." Pattern
    # Manchmal wird is_copula_predicate nicht gesetzt, aber das Pattern ist trotzdem da
    if (
        text_lower.startswith("das sind ")
        or text_lower.startswith("dies sind ")
        or text_lower.startswith("es sind ")
    ):
        if features["is_copula"]:
            return True, "Das/Dies/Es sind-Konstruktion (Kopula-Prädikativ)"

    # Fragesätze sind oft nur eine Frage, nicht mehrere Statements
    # "Was braucht es noch?" = eine Frage
    if features["text"].strip().endswith("?"):
        # Einfache Fragen ohne Konjunktionen = ein Statement
        if not features["has_coord_conjunction"] and not features["has_contrast"]:
            return True, "Einfacher Fragesatz (ein Statement)"

    # Modalverb + Infinitiv (muss abbauen, kann lesen)
    # NUR bei SEHR einfachen Modalverb-Infinitiv ohne jegliche Komplexität
    if features.get("is_modal_infinitive", False):
        # Strikterer Check: NUR wenn KEINE Koordination UND kurzer Satz
        if not features["has_coord_conjunction"] and features["num_tokens"] < 10:
            return True, "Modalverb-Infinitiv-Konstruktion (eine Verbphrase)"

    # Perfekt/Auxiliar-Konstruktion - NUR für SEHR einfache Fälle
    # Nur wenn: Auxiliar + einzelnes Partizip, kurzer Satz, keine Koordination
    if features.get("is_perfect_tense", False):
        # Sehr strikte Bedingungen um False Negatives zu vermeiden
        if (
            not features["has_coord_conjunction"]
            and features["num_tokens"] < 8
            and features["num_commas"] == 0
        ):
            return True, "Einfache Perfekt-Konstruktion (eine Zeitform)"

    # Vergleichskonstruktion (wie, als)
    if features["has_comparison"] and not features["has_coord_conjunction"]:
        return True, "Vergleichskonstruktion (ein Statement)"

    # ======================================================================
    # RULE 1: Strukturelle Multi-Statement Indikatoren
    # ======================================================================
    # ERST NACH den speziellen Konstruktionen prüfen

    if features["num_subjects"] > 1:
        # Mehrere verschiedene Subjekte = sehr wahrscheinlich Multi-Statement
        # (Aber nur wenn keine Kopula-Prädikativ-Korrektur erfolgt ist)
        return False, "Mehrere Subjekte gefunden"

    if features["has_contrast"]:
        # Kontrastive Konjunktion = sehr wahrscheinlich Multi-Statement
        return False, "Kontrastive Konjunktion gefunden"

    # ======================================================================
    # RULE 2: Semantische Kohärenz bei koordinierten Verben
    # ======================================================================
    # "Diese Menschen beobachten und kontrollieren:"
    # "Menschen sollen dabei sein und mitmachen."
    # → Verben aus demselben semantischen Wortfeld = Ein Statement
    #
    # Rationale:
    # - Semantisch verwandte Verben bilden oft eine konzeptuelle Einheit
    # - z.B. "beobachten und kontrollieren" = Überwachungstätigkeit
    # - Linguistisch fundiert, kein Overfitting
    #
    # ABER: Nur bei sehr engen Verben und kurzem Satz anwenden
    # Sonst werden echte Multi-Statements in normaler Sprache übersehen

    if features["semantically_coherent"]:
        # Zusätzliche Bedingungen für normale Sprache:
        # - Verben müssen direkt nebeneinander stehen (< 3 Tokens Distanz)
        # - Oder der Satz muss kurz sein (< 12 Tokens)
        if features.get("verbs_close_together", False) or features["num_tokens"] < 12:
            return True, f"Semantisch kohärente Verben ({features['semantic_field']})"

    # ======================================================================
    # RULE 3: Generelle Kopula-Konstruktionen (hohe Priorität)
    # ======================================================================
    # "Diese Gesetze sind sehr wichtig:"
    # → Kopula + Prädikativ = Eine Eigenschaftsaussage

    if features["is_copula"]:
        return True, "Kopula-Konstruktion (Eigenschaftsaussage)"

    # ======================================================================
    # RULE 4: Expletives "es" Pattern
    # ======================================================================
    # "In einem Museum muss es Informationen geben."
    # → Expletives "es" (dummy subject), KEINE echte Koordination
    #
    # Rationale:
    # - "muss es" / "gibt es" sind idiomatische Konstruktionen
    # - Trainingsdaten haben 0 Beispiele mit diesem Pattern
    # - Sehr häufig in Leichter Sprache (Amtssprache)
    if features["has_expletive_es"]:
        # Prüfe ob KEINE echte Koordination vorliegt
        if not features["has_coord_conjunction"] and num_commas < 2:
            # Extra: Prüfe auf "muss es" / "gibt es" Pattern
            if " muss es " in text_lower or " gibt es " in text_lower:
                return True, "Expletive-es-Pattern (muss/gibt es)"

    # ======================================================================
    # RULE 5: "auch" ohne strukturelle Komplexität
    # ======================================================================
    # "Die Gruppe macht auch Dinge kaputt."
    # → "auch" als Fokuspartikel, KEINE Koordination
    #
    # Rationale:
    # - "auch" allein indiziert KEINE mehreren Statements
    # - Nur problematisch in Kombination mit Koordination
    # - Trainingsdaten: 32.8% mit "auch" haben mehrere Statements
    #   → Aber nur bei Koordination oder Kommata
    if features["has_auch"]:
        # Keine Koordination, keine Kommata, kurzer Satz
        if (
            not features["has_coord_conjunction"]
            and not features["has_coordinated_verbs"]
            and num_commas == 0
            and num_tokens < 12
        ):
            return True, "Auch-Pattern ohne Koordination"

    # ======================================================================
    # RULE 6: Einfache Modalverb-Sätze
    # ======================================================================
    # "Menschen können Informationen verstehen."
    # → Modalverb + Infinitiv, KEINE Koordination
    #
    # Rationale:
    # - Typische Struktur in Leichter Sprache
    # - Nur problematisch bei Koordination oder sehr langen Sätzen
    # - Trainingsdaten-Bias: 47.6% mit "können" haben mehrere Statements
    #   → Aber fast alle haben Koordination oder Kommata
    if features["has_modalverb"]:
        # Keine Koordination, keine Kommata, moderate Länge
        if (
            not features["has_coord_conjunction"]
            and not features["has_coordinated_verbs"]
            and num_commas == 0
            and num_tokens < 15
        ):
            # Extra: Ausnahme für "können ... alle ... alle" (problematisch)
            if text_lower.count(" alle ") < 2:
                return True, "Einfacher Modalverb-Satz"

    # ======================================================================
    # RULE 7: Koordination von Objekten (nicht Verben)
    # ======================================================================
    # "Die Menschen benutzen einen Computer oder ein Tablet."
    # "Besucher können Männer und Frauen sein."
    # → Koordinierte OBJEKTE/NOMEN, NICHT Verben = Ein Statement
    #
    # Rationale (GENERELLES LINGUISTISCHES PRINZIP):
    # - Mehrere Statements = Mehrere PRÄDIKATE (Verben)
    # - Koordinierte Objekte/Nomen = Ein Prädikat, mehrere Argumente
    # - "oder" bei Objekten = Wahlmöglichkeit, kein Multi-Statement
    # - "und" bei Objekten = Aufzählung, kein Multi-Statement
    #
    # WICHTIG: Dies ist KEIN Overfitting, sondern ein fundamentales
    # linguistisches Prinzip in allen Sprachen!
    if features["has_coord_conjunction"]:
        # Koordination vorhanden, aber KEINE koordinierten Verben
        if not features["has_coordinated_verbs"]:
            # Zusätzlich: Nicht zu viele Kommata (kein komplexer Satz)
            if num_commas < 2:
                return True, "Koordination von Objekten (kein Multi-Statement)"

    # ======================================================================
    # RULE 8: Keine strukturelle Komplexität
    # ======================================================================
    # Generische Regel für einfache Sätze ohne jegliche Koordination
    #
    # Rationale:
    # - Keine Konjunktionen, keine koordinierten Verben
    # - Nur 1 Komma oder keines
    # - Kurzer bis mittlerer Satz
    if (
        not features["has_coord_conjunction"]
        and not features["has_coordinated_verbs"]
        and num_commas <= 1
        and num_tokens < 7
    ):
        return True, "Einfache Struktur ohne Koordination"

    # Unsicher → BERT fragen
    return False, None


def _split_sentences_at_punctuation(doc: Doc) -> List[Span]:
    """
    Teilt Sätze zusätzlich an Satzzeichen auf, die spaCy manchmal übersieht.

    SpaCy erkennt normalerweise '.', '!', '?' als Satzenden, aber manchmal
    werden mehrere Frage-/Ausrufesätze fälschlicherweise zusammengefasst.
    Diese Funktion stellt sicher, dass jedes Satzende-Zeichen (:, ?, !)
    auch wirklich einen neuen Satz beginnt.

    Args:
        doc: spaCy Doc-Objekt

    Returns:
        Liste von Span-Objekten (Sätze, inkl. zusätzlicher Splits)
    """
    extended_sents = []

    for sent in doc.sents:
        # Suche nach Satzende-Zeichen im Satz: ':', '?', '!'
        # Diese sollten jeweils einen neuen Satz beginnen
        split_positions = []

        for i, token in enumerate(sent):
            if token.text in [":", "?", "!"]:
                split_positions.append(i)

        if not split_positions:
            # Keine Split-Zeichen → Verwende Original-Satz
            extended_sents.append(sent)
        else:
            # Teile Satz an den gefundenen Zeichen auf
            start_idx = sent.start

            for split_idx in split_positions:
                # Token-Index im Dokument (nicht relativ zum Satz)
                doc_split_idx = sent.start + split_idx

                # Erstelle Sub-Satz bis zum Satzzeichen (inklusive)
                if doc_split_idx + 1 > start_idx:  # Mindestens 1 Token
                    sub_sent = doc[start_idx : doc_split_idx + 1]

                    # Überspringe Sätze die nur aus Whitespace/Newlines bestehen
                    text = sub_sent.text.strip()
                    if text and not text.isspace():
                        extended_sents.append(sub_sent)

                    start_idx = (
                        doc_split_idx + 1
                    )  # Nächster Teil beginnt nach dem Zeichen

            # Rest des Satzes (nach dem letzten Zeichen)
            if start_idx < sent.end:
                sub_sent = doc[start_idx : sent.end]
                text = sub_sent.text.strip()
                if text and not text.isspace():
                    extended_sents.append(sub_sent)

    return extended_sents


def check_rule(doc: Doc) -> List[str]:
    """
    Prüft, ob Sätze mehrere Aussagen/Statements enthalten.

    ÄNDERUNGEN (2025-10-20):
    - Satzzeichen ':', '?', '!' werden jetzt als Satzenden erkannt
    - Stellt sicher, dass mehrere Frage-/Ausrufesätze korrekt getrennt werden
    - Behebt Problem wo spaCy mehrere Sätze zusammenfasst

    ÄNDERUNGEN (2025-10-13 - Option 3 Feature Gating):
    - Nutzt spaCy-basierten Pre-Filter vor BERT
    - Überspringt offensichtliche Ein-Statement-Sätze (schneller + präziser)
    - Ruft BERT nur bei Unsicherheit auf

    Args:
        doc: spaCy Doc-Objekt mit geparsten Sätzen

    Returns:
        Liste von Fehlermeldungen für Sätze mit mehreren Statements
    """
    errors = []

    # Statistics for logging (optional)
    skipped_by_prefilter = 0
    asked_bert = 0

    # WICHTIG: Verwende erweiterte Satzerkennung mit Satzzeichen-Splits
    extended_sents = _split_sentences_at_punctuation(doc)

    for sent in extended_sents:
        sent_text = sent.text.strip()

        # Überspringe sehr kurze Sätze (< 4 Wörter)
        if len(sent_text.split()) < 4:
            continue

        # ===================================================================
        # PRE-FILTER: spaCy-basierte linguistische Analyse
        # ===================================================================
        # Prüfe ob dieser Satz OFFENSICHTLICH nur ein Statement hat
        # → Wenn ja: Skip BERT (schneller + verhindert False Positives)
        # → Wenn nein: BERT fragen (unsichere Fälle)

        is_obvious_single, reason = _is_obvious_single_statement(sent)

        if is_obvious_single:
            # Definitiv ein Statement → Skip BERT
            skipped_by_prefilter += 1
            # Optional: Debug-Logging
            # print(f"[PRE-FILTER] Übersprungen ({reason}): {sent_text[:50]}...")
            continue

        # ===================================================================
        # BERT-INFERENCE: Nur bei unsicheren Fällen
        # ===================================================================
        asked_bert += 1
        has_multiple, confidence = _predict_multiple_statements(sent, doc)

        if has_multiple:
            # Erstelle aussagekräftige Fehlermeldung
            if confidence >= 0.8:
                sicherheit = "sehr wahrscheinlich"
            elif confidence >= 0.6:
                sicherheit = "wahrscheinlich"
            else:
                sicherheit = "möglicherweise"

            # Formatiere Satz-Anzeige basierend auf Konfiguration
            if SHOW_FULL_SENTENCE:
                # Zeige vollständigen Satz
                satz_anzeige = f'"{sent_text}"'
            else:
                # Zeige nur erste 80 Zeichen + "..."
                if len(sent_text) > 80:
                    satz_anzeige = f'"{sent_text[:80]}..."'
                else:
                    satz_anzeige = f'"{sent_text}"'

            errors.append(
                f"Dieser Satz enthält {sicherheit} mehrere Aussagen: {satz_anzeige} "
                f"Besser: Teilen Sie den Satz in mehrere kurze Sätze auf. "
                f"Jeder Satz sollte nur eine Aussage enthalten."
            )

    return errors


# Für Kompatibilität mit anderen Regeln
def get_rule_info() -> dict:
    """Gibt Informationen über diese Regel zurück."""
    return {
        "name": "Mehrere Aussagen",
        "description": "Erkennt Sätze mit mehreren Aussagen/Statements (basierend auf StaGE-Dataset)",
        "model_based": True,
        "model_type": "BERT (bert-base-german-cased)",
        "source": "https://german-easy-to-read.github.io/statements/",
        "recommendation": "Ein Satz sollte nur eine Aussage enthalten",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    """Test der Regel."""
    import spacy

    logger.info("Test: Regel zur Erkennung mehrerer Aussagen")
    logger.info("=" * 60)

    # Lade spaCy Modell
    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        logger.error("Deutsches spaCy-Modell nicht verfügbar.")
        logger.error("Installieren mit: python -m spacy download de_core_news_lg")
        exit(1)

    # Test-Texte
    test_texts = [
        "Das Haus ist groß.",  # 1 Statement - KEIN Fehler
        "Das Haus ist groß und hat einen schönen Garten.",  # 2 Statements - FEHLER
        "Der Mann geht in den Park und kauft ein Eis.",  # 2 Statements - FEHLER
        "Es regnet heute.",  # 1 Statement - KEIN Fehler
        "Maria arbeitet im Büro, sie ist sehr fleißig und macht ihre Arbeit gut.",  # 3 Statements - FEHLER
        "Der Hund bellt laut.",  # 1 Statement - KEIN Fehler
        "Die Frau kocht Essen, während der Mann fernsieht.",  # 2 Statements - FEHLER
        "Ich gehe einkaufen.",  # 1 Statement - KEIN Fehler
    ]

    for i, text in enumerate(test_texts, 1):
        logger.info('\n%s. Text: "%s"', i, text)

        doc = nlp(text)
        results = check_rule(doc)

        if results:
            logger.error("   FEHLER gefunden:")
            for error in results:
                logger.error("      %s", error)
        else:
            logger.info("   OK: Keine mehreren Aussagen erkannt")

    logger.info("\n" + "=" * 60)
    logger.info("Test abgeschlossen!")

    # Zeige Regel-Info
    info = get_rule_info()
    logger.info("\nRegel-Information:")
    for key, value in info.items():
        logger.info("   %s: %s", key, value)
