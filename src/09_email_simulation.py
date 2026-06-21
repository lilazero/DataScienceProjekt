"""
09_email_simulation.py

Aufgabenbereich: Deep Learning, Automatisierung
"Automatisierte Verarbeitung von Kundeninformationen aus E-Mails oder Texten"

WICHTIGER HINWEIS (siehe Aufgabendefinition, Folie 14):
  - Es wird AUSSCHLIESSLICH eine Test-E-Mail mit fiktiven Daten erzeugt.
  - Es werden KEINE echten Kundendaten verarbeitet.
  - Dieses Skript dient als einfache Simulation/Emulation einer eingehenden
    Kundenanfrage, wie sie z.B. über ein Kontaktformular oder per E-Mail
    eintreffen könnte. Eine echte Verarbeitung realer E-Mails (inkl.
    Anonymisierung, Datenschutzmaßnahmen) liegt außerhalb des Projektumfangs.

Das Skript erstellt eine Textdatei (.txt), die wie eine typische
Kundenanfrage aussieht: Ein (fiktiver) Interessent beschreibt eine
Immobilie und fragt nach einer Preisschätzung.
"""

import random
from datetime import datetime

from config import OUTPUT_DIR

EMAIL_OUTPUT_PATH = f"{OUTPUT_DIR}/beispiel_kundenanfrage.txt"

# Fiktive Testdaten - KEINE echten Kundendaten oder echte Personen.
TEST_LOCATIONS = ["Whitefield", "Electronic City", "Hebbal", "Indiranagar", "Marathahalli"]
TEST_AREA_TYPES = ["Super built-up  Area", "Built-up  Area", "Plot  Area"]
TEST_SENDER_NAMES = ["Anil Kumar", "Priya Sharma", "Rohit Mehta", "Sunita Rao"]


def generate_test_inquiry(seed: int = 42) -> dict:
    """
    Generiert fiktive Beispieldaten fuer eine Kundenanfrage.
    Diese Daten sind frei erfunden (Testdaten) und stammen nicht von einer
    echten Person.
    """
    rng = random.Random(seed)
    return {
        "sender_name": rng.choice(TEST_SENDER_NAMES),
        "sender_email": "test.kunde@beispiel-mail.invalid",
        "location": rng.choice(TEST_LOCATIONS),
        "area_type": rng.choice(TEST_AREA_TYPES),
        "total_sqft": rng.choice([1000, 1200, 1500, 1800, 2200]),
        "bhk": rng.choice([2, 3, 4]),
        "bath": rng.choice([2, 3]),
        "balcony": rng.choice([1, 2]),
    }


def build_email_text(data: dict) -> str:
    """Formatiert die Testdaten als realistisch wirkende E-Mail (reiner Text)."""
    timestamp = datetime.now().strftime("%d.%m.%Y, %H:%M Uhr")

    return f"""Von: {data['sender_name']} <{data['sender_email']}>
An: info@yay-studios-real-estate.invalid
Betreff: Preisanfrage fuer Immobilie in {data['location']}
Datum: {timestamp}

Sehr geehrtes Team,

ich interessiere mich fuer eine Immobilie und wuerde gerne eine
Preiseinschaetzung erhalten. Die Eckdaten sind wie folgt:

  - Lage: {data['location']}
  - Flaechenart: {data['area_type']}
  - Wohnflaeche: {data['total_sqft']} sqft
  - Zimmeranzahl (BHK): {data['bhk']}
  - Anzahl Badezimmer: {data['bath']}
  - Anzahl Balkone: {data['balcony']}

Koennten Sie mir bitte mitteilen, mit welchem ungefaehren Kaufpreis ich fuer
eine solche Immobilie rechnen muesste?

Vielen Dank im Voraus fuer Ihre Rueckmeldung.

Mit freundlichen Gruessen
{data['sender_name']}


---
HINWEIS: Dies ist eine simulierte Test-E-Mail mit frei erfundenen Daten.
Es handelt sich nicht um eine echte Kundenanfrage und es wurden keine
echten Personen- oder Kundendaten verwendet (siehe Datenschutz-Hinweis
in der Projektdokumentation).
"""


def main():
    print("=== Schritt 9: Beispiel-Kundenanfrage (E-Mail-Simulation) ===\n")
    print("Hinweis: Es werden ausschliesslich Testdaten verwendet, keine")
    print("echten Kundendaten (siehe Aufgabendefinition, 'Wichtiger Hinweis').\n")

    data = generate_test_inquiry()
    email_text = build_email_text(data)

    with open(EMAIL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(email_text)

    print(f"Beispiel-E-Mail erstellt: {EMAIL_OUTPUT_PATH}\n")
    print("--- Inhalt der erzeugten Test-E-Mail ---\n")
    print(email_text)


if __name__ == "__main__":
    main()
