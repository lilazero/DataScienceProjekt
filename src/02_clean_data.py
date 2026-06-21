"""
02_clean_data.py

Deskriptive Statistik und Datenverständnis (Teil 1):
"Identifikation von Ausreißern und fehlenden Werten", "Bewertung der
Datenqualität".

Dieses Skript:
  1. Lädt die Rohdaten aus MySQL (so wie sie in Schritt 1 gespeichert wurden)
  2. Bereinigt total_sqft (Wertebereiche, Flächeneinheiten -> sqft)
  3. Extrahiert die BHK-Zahl (Zimmeranzahl) aus der Spalte 'size'
  4. Behandelt fehlende Werte
  5. Entfernt offensichtliche Datenfehler / extreme Ausreißer
  6. Erstellt die abgeleitete Kennzahl price_per_sqft
  7. Speichert den bereinigten Datensatz als CSV für die weiteren Skripte

Wichtig: Die Originaldaten in MySQL bleiben unverändert (Audit-Trail).
Die bereinigten Daten werden separat als Datei abgelegt und in den
folgenden Analyseschritten verwendet.
"""

import re
import pandas as pd
import numpy as np

from config import OUTPUT_DIR, TABLE_NAME
from db import get_engine

CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"

# Umrechnungsfaktoren verschiedener in Indien gebräuchlicher Flächeneinheiten
# in Quadratfuß (sqft), wie sie im Rohdatensatz vorkommen.
UNIT_TO_SQFT = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Acres": 43560.0,
    "Cents": 435.6,
    "Grounds": 2400.0,
    "Guntha": 1089.0,
    "Perch": 272.25,
}


def load_from_mysql() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", con=engine)
    print(f"📄 {len(df)} Zeilen aus MySQL geladen.")
    return df


def convert_sqft(value) -> float:
    """
    Wandelt verschiedene total_sqft-Formate in eine einheitliche
    Quadratfuß-Zahl um:
      - reine Zahl            "1056"            -> 1056.0
      - Wertebereich          "2100 - 2850"      -> Mittelwert (2475.0)
      - alternative Einheit   "34.46Sq. Meter"   -> in sqft umgerechnet
    Nicht interpretierbare Werte werden zu NaN (werden später entfernt).
    """
    s = str(value).strip()

    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                a, b = float(parts[0].strip()), float(parts[1].strip())
                return (a + b) / 2
            except ValueError:
                return np.nan

    for unit, factor in UNIT_TO_SQFT.items():
        if unit in s:
            num_part = s.replace(unit, "").strip()
            try:
                return float(num_part) * factor
            except ValueError:
                return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_bhk(size_value) -> float:
    """Extrahiert die Zimmeranzahl aus Werten wie '2 BHK', '4 Bedroom', '1 RK'."""
    if pd.isna(size_value):
        return np.nan
    match = re.search(r"(\d+)", str(size_value))
    return float(match.group(1)) if match else np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_start = len(df)
    report = {}

    # --- 1. total_sqft bereinigen -------------------------------------
    df["total_sqft_clean"] = df["total_sqft"].apply(convert_sqft)
    report["total_sqft_unparsable"] = df["total_sqft_clean"].isna().sum()
    df = df.dropna(subset=["total_sqft_clean"])

    # --- 2. BHK aus 'size' extrahieren ---------------------------------
    df["bhk"] = df["size"].apply(extract_bhk)
    report["size_missing_or_unparsable"] = df["bhk"].isna().sum()
    df = df.dropna(subset=["bhk"])
    df["bhk"] = df["bhk"].astype(int)

    # --- 3. Fehlende Werte behandeln -----------------------------------
    # location: die 1 fehlende Zeile entfernen (zu wenige Fälle für Imputation)
    report["location_missing_dropped"] = df["location"].isna().sum()
    df = df.dropna(subset=["location"])

    # society: sehr viele fehlende Werte (>40%) -> als 'Unknown' kennzeichnen
    # statt die Zeilen zu verwerfen, da society für die Kernanalyse nicht
    # zwingend benötigt wird.
    df["society"] = df["society"].fillna("Unknown")

    # bath/balcony: fehlende Werte mit dem Median der jeweiligen BHK-Gruppe auffüllen,
    # da die Anzahl der Bäder/Balkone typischerweise mit der Wohnungsgröße zusammenhängt.
    report["bath_missing_imputed"] = df["bath"].isna().sum()
    df["bath"] = df.groupby("bhk")["bath"].transform(lambda x: x.fillna(x.median()))
    df["bath"] = df["bath"].fillna(df["bath"].median())

    report["balcony_missing_imputed"] = df["balcony"].isna().sum()
    df["balcony"] = df.groupby("bhk")["balcony"].transform(lambda x: x.fillna(x.median()))
    df["balcony"] = df["balcony"].fillna(df["balcony"].median())

    # --- 4. Abgeleitete Kennzahl: Preis pro Quadratfuß ------------------
    # price liegt in Lakhs (1 Lakh = 100.000 INR) vor.
    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft_clean"]

    # --- 5. Plausibilitätsprüfung / Ausreißerbehandlung ------------------
    # a) Offensichtliche Dateneingabefehler bei total_sqft: Werte, die nach der
    #    Einheitenumrechnung > 20.000 sqft ergeben (~1858 m²), sind für Wohnungen
    #    in diesem Datensatz unrealistisch (z.B. "30Acres" für eine "2 BHK"-Wohnung,
    #    vermutlich fälschlich als Flächenangabe statt als Grundstücksgröße erfasst).
    before = len(df)
    df = df[df["total_sqft_clean"] <= 20000]
    report["total_sqft_implausible_removed"] = before - len(df)

    # c) Unrealistisches Verhältnis sqft/BHK: weniger als 300 sqft pro Zimmer
    #    deutet auf Dateneingabefehler hin (z.B. "43 BHK" auf kleiner Fläche).
    before = len(df)
    df = df[df["total_sqft_clean"] / df["bhk"] >= 300]
    report["unrealistic_sqft_per_bhk_removed"] = before - len(df)

    # d) Extreme Ausreißer bei price_per_sqft mittels IQR-Methode entfernen
    #    (innerhalb jeder Location, da sich Preise je Stadtteil stark unterscheiden)
    before = len(df)

    def pps_bounds(group):
        q1, q3 = group.quantile([0.25, 0.75])
        iqr = q3 - q1
        return pd.Series({"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr})

    bounds = df.groupby("location")["price_per_sqft"].apply(pps_bounds).unstack()
    df = df.join(bounds, on="location")
    df = df[(df["price_per_sqft"] >= df["lower"]) & (df["price_per_sqft"] <= df["upper"])]
    df = df.drop(columns=["lower", "upper"])
    report["price_per_sqft_outliers_removed"] = before - len(df)

    # e) bath darf realistisch nicht deutlich höher sein als bhk + 2
    before = len(df)
    df = df[df["bath"] <= df["bhk"] + 2]
    report["unrealistic_bath_count_removed"] = before - len(df)

    # f) Vollständige Duplikate entfernen (gleiche Werte in allen Spalten)
    before = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = before - len(df)

    # --- 6. Seltene Locations zusammenfassen -----------------------------
    # Bei 1300+ unterschiedlichen Locations enthalten viele nur 1-2 Einträge.
    # Für die Modellierung werden Locations mit < 10 Einträgen zu "other"
    # zusammengefasst, um Overfitting bei One-Hot-Encoding zu vermeiden.
    df["location"] = df["location"].str.strip()
    location_counts = df["location"].value_counts()
    rare_locations = location_counts[location_counts < 10].index
    df["location_grouped"] = df["location"].apply(
        lambda x: "other" if x in rare_locations else x
    )
    report["rare_locations_grouped_to_other"] = len(rare_locations)

    df = df.reset_index(drop=True)

    print("\n--- Datenbereinigungs-Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nZeilen vorher: {n_start}  ->  Zeilen nachher: {len(df)} "
          f"({n_start - len(df)} entfernt, {100 * (n_start - len(df)) / n_start:.1f}%)")

    return df


def main():
    print("=== Schritt 2: Daten bereinigen ===\n")
    df_raw = load_from_mysql()
    df_clean = clean_data(df_raw)

    df_clean.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"\n✅ Bereinigte Daten gespeichert unter: {CLEAN_CSV_PATH}")
    print(f"\nSpalten im bereinigten Datensatz:\n{list(df_clean.columns)}")
    print(f"\nVorschau:\n{df_clean.head()}")


if __name__ == "__main__":
    main()
