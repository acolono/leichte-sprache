"""Configuration for the komposita (compound word) rule."""

from typing import Dict, List, Set

# Detection modes
MODE_ALL_COMPOUNDS = "alle"
MODE_BY_LENGTH = "laenge"
MODE_BY_PARTS = "teile"
MODE_COMBINED = "kombiniert"

# Active detection mode
MODE = MODE_COMBINED

# Minimum word length for BY_LENGTH mode
MIN_WORD_LENGTH = 12

# Minimum component count for BY_PARTS mode
MIN_PART_COUNT = 3

# Absolute minimum length (words shorter than this are never compounds)
MIN_COMPOUND_LENGTH = 6

# Word frequency threshold
WORD_FREQ_THRESHOLD = 2e-7

# Language setting
LANGUAGE = "de"

# Linking elements in German compounds
LINKING_SOUNDS = ["s", "es", "n", "en", "er"]

# Known compound words dictionary
KNOWN_COMPOUNDS: Dict[str, List[str]] = {
    # 3+ Teile Komposita (komplexe Komposita)
    "festplattenrekorder": ["fest", "platte", "rekorder"],
    "bundespräsidentenstichwahl": ["bund", "präsident", "stich", "wahl"],
    "datenschutzgrundverordnung": ["daten", "schutz", "grund", "verordnung"],
    "projektentwicklungsleiter": ["projekt", "entwicklung", "leiter"],
    "softwareentwicklungsrichtlinie": ["software", "entwicklung", "richtlinie"],
    "qualitätssicherungsabteilung": ["qualität", "sicherung", "abteilung"],
    "benutzerfreundlichkeitsaspekt": ["benutzer", "freundlichkeit", "aspekt"],
    "grundstücksverkehrsgenehmigung": ["grundstück", "verkehr", "genehmigung"],
    "kraftfahrzeughaftpflichtversicherung": [
        "kraftfahrzeug",
        "haftpflicht",
        "versicherung",
    ],
    "kundenbeziehungsmanagement": ["kunden", "beziehung", "management"],
    "geschäftsprozessoptimierung": ["geschäft", "prozess", "optimierung"],
    # Problematische Fälle aus dem Output - mit korrekten Zerlegungen
    "bezirksverwaltungsamt": ["bezirk", "verwaltung", "amt"],
    "grundstücksgrenzenbebauungsverordnung": [
        "grundstück",
        "grenze",
        "bebauung",
        "verordnung",
    ],
    "lärmschutzgutachtens": ["lärm", "schutz", "gutachten"],
    "nummernausgabeautomaten": ["nummer", "ausgabe", "automat"],
    "digitalanzeigetafel": ["digital", "anzeige", "tafel"],
    "oberlippenbart": ["ober", "lippe", "bart"],
    "dienstleistungsstimme": ["dienst", "leistung", "stimme"],
    # Weitere problematische Fälle
    "herbstnachmittag": ["herbst", "nachmittag"],
    "werkzeugaufbewahrung": ["werkzeug", "aufbewahrung"],
    "sondergenehmigungsverfahren": ["sonder", "genehmigung", "verfahren"],
    "materialbeständigkeitsnachweises": ["material", "beständigkeit", "nachweis"],
    "fassungslosigkeit": ["fassung", "losigkeit"],
    "serverüberlastung": ["server", "überlastung"],
    "fahrscheinautomaten": ["fahrschein", "automat"],
    "geistesabwesenheit": ["geist", "abwesenheit"],
    "nachkriegsarchitektur": ["nachkrieg", "architektur"],
    "zuständigkeitsbereich": ["zuständigkeit", "bereich"],
    "bürgerberatungszentrum": ["bürger", "beratung", "zentrum"],
    "informationsbroschüre": ["information", "broschüre"],
    "antragsformularstapel": ["antrag", "formular", "stapel"],
    # Problematische Fälle aus neuem Output - korrekte Zerlegungen
    "bauabnahmeabteilung": ["bau", "abnahme", "abteilung"],
    "nervenzusammenbruchs": ["nerven", "zusammen", "bruch"],
    "verwaltungsvorschriften": ["verwaltung", "vorschrift"],
    "bürokratiemonster": ["bürokratie", "monster"],
    "formularberg": ["formular", "berg"],
    "windmühlenflügel": ["windmühle", "flügel"],
    "gartenhausglück": ["garten", "haus", "glück"],
    # Zusätzliche problematische Fälle - verhindert kaputte Zerlegungen
    "unternehmensstrategie": ["unternehmen", "strategie"],
    "softwareentwicklung": ["software", "entwicklung"],
    "kundenbetreuungsservice": ["kunden", "betreuung", "service"],
    "projektmanagementtools": ["projekt", "management", "tools"],
    "qualitätskontrollsystem": ["qualität", "kontrolle", "system"],
    "datenverarbeitungsanlage": ["daten", "verarbeitung", "anlage"],
    "kommunikationsstrategie": ["kommunikation", "strategie"],
    "geschäftsführungsebene": ["geschäftsführung", "ebene"],
    "informationsübertragung": ["information", "übertragung"],
    "sicherheitsbestimmungen": ["sicherheit", "bestimmung"],
    "arbeitsschutzvorschriften": ["arbeitsschutz", "vorschrift"],
    "umweltschutzmaßnahmen": ["umweltschutz", "maßnahme"],
    "verkehrssicherheitskampagne": ["verkehr", "sicherheit", "kampagne"],
    "bildungseinrichtungen": ["bildung", "einrichtung"],
    "forschungsinstitut": ["forschung", "institut"],
    "entwicklungsabteilung": ["entwicklung", "abteilung"],
    "personalverwaltung": ["personal", "verwaltung"],
    "finanzdienstleistung": ["finanz", "dienstleistung"],
    "marketingkampagne": ["marketing", "kampagne"],
    "vertriebsorganisation": ["vertrieb", "organisation"],
    # 2 Teile Komposita (normale Komposita - nicht komplex)
    "haustür": ["haus", "tür"],
    "regierungskommission": ["regierung", "kommission"],
    "verwaltungsverfahren": ["verwaltung", "verfahren"],
    "krankenversicherung": ["kranken", "versicherung"],
    "arbeitsplatzsicherheit": ["arbeitsplatz", "sicherheit"],
    "umweltschutzgesetz": ["umweltschutz", "gesetz"],
    "bundestagsabgeordneter": ["bundestag", "abgeordneter"],
    "sozialversicherungsbeitrag": ["sozialversicherung", "beitrag"],
    "einkommenssteuergesetz": ["einkommensteuer", "gesetz"],
    "betriebsvereinbarung": ["betrieb", "vereinbarung"],
    "marktforschungsunternehmen": ["marktforschung", "unternehmen"],
    "informationstechnologie": ["information", "technologie"],
    "telekommunikationsanbieter": ["telekommunikation", "anbieter"],
    "energieversorgungsunternehmen": ["energieversorgung", "unternehmen"],
    "verkehrsinfrastruktur": ["verkehr", "infrastruktur"],
    "bildungseinrichtung": ["bildung", "einrichtung"],
    "gesundheitsversorgung": ["gesundheit", "versorgung"],
    "wirtschaftswachstum": ["wirtschaft", "wachstum"],
    "innovationsmanagement": ["innovation", "management"],
    "qualitätsmanagement": ["qualität", "management"],
    "personalentwicklung": ["personal", "entwicklung"],
}

# Set of known compound words for quick lookup
OWN_DICTIONARY: Set[str] = set(KNOWN_COMPOUNDS.keys())
