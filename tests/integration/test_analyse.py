"""Integration tests for /analyse endpoint -- smoke tests and domain-specific validation."""
import pytest


def assert_analyse_response(response, expected_rules: set):
    """Validate /analyse response schema and assert minimum rule set fires.

    Args:
        response: HTTP response from /analyse endpoint.
        expected_rules: Set of rule names that must appear in violations_by_rule.
    """
    assert response.status_code == 200
    data = response.json()
    assert "annotated_text" in data
    assert "statistics" in data
    assert "issues" in data
    assert "violations_by_rule" in data["statistics"]
    fired_rules = set(data["statistics"]["violations_by_rule"].keys())
    missing = expected_rules - fired_rules
    assert not missing, (
        f"Expected rules did not fire: {missing}. "
        f"Fired rules: {sorted(fired_rules)}"
    )


@pytest.mark.integration
class TestAnalyseSmoke:
    """Minimal smoke tests to verify TestClient and fixtures."""

    def test_health_endpoint(self, client):
        """Verify /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_analyse_returns_valid_structure(self, client):
        """Verify /analyse returns expected JSON keys."""
        response = client.post("/analyse", json={"text": "Das ist ein Test."})
        assert response.status_code == 200
        data = response.json()
        assert "annotated_text" in data
        assert "statistics" in data
        assert "issues" in data
        assert "violations_by_rule" in data["statistics"]


@pytest.mark.integration
class TestAnalyseGovernment:
    """Government domain (Verwaltungsbescheid) integration tests."""

    GOVERNMENT_TEXT = (
        "Gemäß der Verwaltungsvorschrift des Bundesministeriums für Arbeit und "
        "Soziales wird hiermit festgestellt, dass der Antragsteller die "
        "Voraussetzungen für die Bewilligung der beantragten Sozialleistungen "
        "nicht erfüllt hat. Die Ablehnung des Antrags erfolgte aufgrund der "
        "unvollständigen Dokumentation, welche trotz mehrfacher Aufforderung "
        "nicht nachgereicht wurde. Es wird darauf hingewiesen, dass gegen "
        "diesen Bescheid innerhalb eines Monats nach Zustellung Widerspruch "
        "eingelegt werden kann. Der Widerspruch ist schriftlich oder zur "
        "Niederschrift bei der zuständigen Behörde einzureichen."
    )

    def test_government_text_triggers_expected_rules(self, client):
        """Government text triggers satzlaenge, passiv_erkennung, genitiv, komposita."""
        response = client.post("/analyse", json={"text": self.GOVERNMENT_TEXT})
        assert_analyse_response(
            response, {"satzlaenge", "passiv_erkennung", "genitiv", "komposita"}
        )


@pytest.mark.integration
class TestAnalyseHealth:
    """Health domain (Beipackzettel) integration tests."""

    HEALTH_TEXT = (
        "Dieses Arzneimittel enthält den Wirkstoff Acetylsalicylsäure, der "
        "zur symptomatischen Behandlung von leichten bis mäßig starken "
        "Schmerzen und Fieber angewendet wird. Nehmen Sie das Medikament "
        "nicht ein, wenn Sie eine Überempfindlichkeit gegen nichtsteroidale "
        "Antirheumatika haben oder an einer Blutgerinnungsstörung leiden. "
        "Häufige Nebenwirkungen sind gastrointestinale Beschwerden wie "
        "Übelkeit, Erbrechen und Magenschmerzen. Bei Auftreten von "
        "allergischen Reaktionen ist die Einnahme sofort abzubrechen und "
        "ein Arzt zu konsultieren."
    )

    def test_health_text_triggers_expected_rules(self, client):
        """Health text triggers kurze_woerter, komposita, satzlaenge, fremdwoerter."""
        response = client.post("/analyse", json={"text": self.HEALTH_TEXT})
        assert_analyse_response(
            response, {"kurze_woerter", "komposita", "satzlaenge", "fremdwoerter"}
        )


@pytest.mark.integration
class TestAnalyseEducation:
    """Education domain (Schulordnung) integration tests."""

    EDUCATION_TEXT = (
        "Die Teilnahme an der Ganztagsbetreuung setzt eine rechtzeitige "
        "Anmeldung durch die Erziehungsberechtigten voraus, wobei die "
        "Anmeldefrist jeweils zum Ende des vorherigen Schulhalbjahres "
        "abläuft. Die Schülerinnen und Schüler sind verpflichtet, die "
        "Hausordnung einzuhalten und den Anweisungen des pädagogischen "
        "Personals Folge zu leisten. Bei wiederholten Verstößen gegen die "
        "Schulordnung können disziplinarische Maßnahmen ergriffen werden, "
        "die von einem Verweis bis zum vorübergehenden Ausschluss vom "
        "Unterricht reichen. Die Erziehungsberechtigten werden über "
        "Ordnungsmaßnahmen schriftlich informiert."
    )

    def test_education_text_triggers_expected_rules(self, client):
        """Education text triggers komposita, komplexitaet, nebensaetze, satzlaenge."""
        response = client.post("/analyse", json={"text": self.EDUCATION_TEXT})
        assert_analyse_response(
            response, {"komposita", "komplexitaet", "nebensaetze", "satzlaenge"}
        )


@pytest.mark.integration
class TestAnalyseNews:
    """News domain (Pressemitteilung) integration tests."""

    NEWS_TEXT = (
        "Die Bundesregierung hat am Mittwoch ein umfassendes Konjunkturpaket "
        "in Höhe von 50 Milliarden Euro beschlossen, das die wirtschaftliche "
        "Erholung nach der Rezession beschleunigen soll. Wie der "
        "Regierungssprecher mitteilte, werden die Mittel schwerpunktmäßig "
        "für Infrastrukturprojekte und die Digitalisierung bereitgestellt. "
        "Kritiker bemängelten jedoch, dass das Paket nicht ausreichend auf "
        "die Bedürfnisse des Mittelstands eingehe und zu bürokratisch "
        "ausgestaltet sei. Die Opposition forderte darüber hinaus eine "
        "stärkere Entlastung der Bürgerinnen und Bürger durch "
        "Steuersenkungen."
    )

    def test_news_text_triggers_expected_rules(self, client):
        """News text triggers fremdwoerter, konjunktiv, satzlaenge, komposita."""
        response = client.post("/analyse", json={"text": self.NEWS_TEXT})
        assert_analyse_response(
            response, {"fremdwoerter", "konjunktiv", "satzlaenge", "komposita"}
        )


@pytest.mark.integration
class TestAnalyseTechnical:
    """Technical domain (Bedienungsanleitung) integration tests."""

    TECHNICAL_TEXT = (
        "Vor der Erstinbetriebnahme des Hochdruckreinigungsgeräts muss der "
        "Wasseranschlussadapter auf den Geräteanschlussstutzen aufgeschraubt "
        "werden. Stellen Sie sicher, dass die Stromversorgung den technischen "
        "Spezifikationen entspricht: 230V Wechselspannung, 50Hz, mindestens "
        "16A Absicherung. Die Hochdruckschlauchverbindung darf nicht geknickt "
        "oder über scharfe Kanten geführt werden, da dies zu einem "
        "Druckverlust von ca. 15% führen kann. Bei Fehlfunktionen des "
        "Sicherheitsventils ist das Gerät unverzüglich außer Betrieb zu "
        "nehmen und der Kundendienst zu kontaktieren."
    )

    def test_technical_text_triggers_expected_rules(self, client):
        """Technical text triggers abkuerzungen, zahlwoerter, komposita, satzlaenge."""
        response = client.post("/analyse", json={"text": self.TECHNICAL_TEXT})
        assert_analyse_response(
            response, {"abkuerzungen", "zahlwoerter", "komposita", "satzlaenge"}
        )
