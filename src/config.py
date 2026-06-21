"""
Zentrale Konfiguration für das Projekt.
Passen Sie hier die MySQL-Zugangsdaten an Ihre lokale Umgebung an.
"""

import os

# --- MySQL-Verbindungsdaten -------------------------------------------------
# Empfehlung: Passwort NICHT im Code, sondern als Umgebungsvariable setzen:
#   export DB_PASSWORD="ihr_passwort"      (Linux/Mac)
#   setx DB_PASSWORD "ihr_passwort"        (Windows)
# Falls keine Umgebungsvariable gesetzt ist, wird der Default-Wert unten benutzt.

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": os.environ.get("DB_PASSWORD", "DEIN_PASSWORT_HIER"),
    "database": "yay_studios_real_estate",
}

# --- Pfade -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SQL_DIR = os.path.join(PROJECT_ROOT, "sql")

RAW_CSV_PATH = os.path.join(DATA_DIR, "Bengaluru_House_Data.csv")

TABLE_NAME = "property_listings"

# Zielvariable für das gesamte Projekt
TARGET_VARIABLE = "price"

os.makedirs(OUTPUT_DIR, exist_ok=True)
