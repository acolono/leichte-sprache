"""
Konfiguration für die Abkürzungen (BERT)-Regel.
"""

# Mindest-Konfidenz für das BERT-Modell.
# Nur Abkürzungen mit einer Konfidenz >= diesem Wert werden gemeldet.
# Empfohlene Werte:
#   - 0.80: Filtert unsichere Vorhersagen, erkennt aber Dr., Prof. etc. (empfohlen)
#   - 0.85: Striktere Filterung
#   - 0.90: Sehr strikt, nur hochkonfidente Vorhersagen
MIN_CONFIDENCE_THRESHOLD = 0.80
