"""
01_load_data.py

Aufgabe (laut Aufgabendefinition):
"Speicherung des gesamten Datensatzes zunächst in MySQL"

Dieses Skript liest die Rohdaten aus der CSV-Datei und speichert sie
UNVERÄNDERT in der MySQL-Tabelle `property_listings`. Die eigentliche
Datenbereinigung (z.B. total_sqft) erfolgt bewusst NICHT hier, sondern
erst im nächsten Schritt (02_clean_data.py), damit die Datenbank zunächst
den Originaldatensatz 1:1 widerspiegelt.
"""

import sys
import pandas as pd

from config import RAW_CSV_PATH, TABLE_NAME
from db import get_engine, test_connection


def load_raw_csv() -> pd.DataFrame:
    """Liest die Rohdaten ein, ohne sie inhaltlich zu verändern."""
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"📄 CSV geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


def insert_into_mysql(df: pd.DataFrame) -> None:
    """Schreibt den DataFrame in die bestehende MySQL-Tabelle."""
    engine = get_engine()

    # bath/balcony können NaN enthalten -> als nullable Integer behandeln,
    # damit MySQL NULL statt einer Fließkommazahl erhält.
    df = df.copy()
    df["bath"] = df["bath"].astype("Int64")
    df["balcony"] = df["balcony"].astype("Int64")

    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="append",   # Tabelle existiert bereits laut schema.sql
        index=False,
        chunksize=1000,
        method="multi",
    )
    print(f"✅ {len(df)} Zeilen in Tabelle '{TABLE_NAME}' eingefügt.")


def main():
    print("=== Schritt 1: Rohdaten in MySQL speichern ===\n")

    if not test_connection():
        print("\nAbbruch: Bitte zuerst die Datenbankverbindung in src/config.py prüfen.")
        sys.exit(1)

    df = load_raw_csv()
    insert_into_mysql(df)

    print("\nFertig. Die Rohdaten liegen jetzt vollständig in MySQL vor.")


if __name__ == "__main__":
    main()
