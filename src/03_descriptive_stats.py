"""
03_descriptive_stats.py

Aufgabenbereich: Deskriptive Statistik und Datenverständnis

  - Berechnung von Lageparametern (Mittelwert, Median, Quartile)
  - Berechnung von Streuungsmaßen (Standardabweichung, Varianz)
  - Analyse der Verteilungen der Variablen
  - Identifikation von Ausreißern und fehlenden Werten
  - Bewertung der Datenqualität
  - Erstellung geeigneter Visualisierungen
  - Interpretation und Dokumentation der Ergebnisse

Alle Diagramme werden als PNG-Dateien in /output gespeichert, damit sie
direkt in die Präsentation bzw. Projektdokumentation übernommen werden
können.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # kein interaktives Fenster nötig
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR

sns.set_theme(style="whitegrid")
CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"

NUMERIC_COLS = ["total_sqft_clean", "bath", "balcony", "bhk", "price", "price_per_sqft"]


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV_PATH)
    print(f"📄 Bereinigte Daten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


def lageparameter_und_streuung(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Mittelwert, Median, Quartile, Std.abw. und Varianz."""
    stats = pd.DataFrame({
        "mean": df[NUMERIC_COLS].mean(),
        "median": df[NUMERIC_COLS].median(),
        "q1_25%": df[NUMERIC_COLS].quantile(0.25),
        "q3_75%": df[NUMERIC_COLS].quantile(0.75),
        "std": df[NUMERIC_COLS].std(),
        "variance": df[NUMERIC_COLS].var(),
        "min": df[NUMERIC_COLS].min(),
        "max": df[NUMERIC_COLS].max(),
        "skewness": df[NUMERIC_COLS].skew(),
    })
    return stats


def datenqualitaet_report(df: pd.DataFrame) -> None:
    print("\n--- Datenqualitäts-Übersicht (nach Bereinigung) ---")
    print(f"Anzahl Zeilen: {len(df)}")
    print(f"Anzahl Spalten: {df.shape[1]}")
    print(f"\nFehlende Werte je Spalte:\n{df.isnull().sum()}")
    print(f"\nDuplikate: {df.duplicated().sum()}")
    print(f"\nEindeutige Locations: {df['location'].nunique()}")
    print(f"Eindeutige Location-Gruppen (nach Zusammenfassung seltener Werte): "
          f"{df['location_grouped'].nunique()}")


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogramme der wichtigsten numerischen Variablen."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cols = NUMERIC_COLS
    for ax, col in zip(axes.flat, cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB")
        ax.set_title(f"Verteilung: {col}")
        ax.set_xlabel(col)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_verteilungen_histogramme.png", dpi=150)
    plt.close()
    print(f"📊 Gespeichert: 01_verteilungen_histogramme.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Boxplots zur Identifikation von Ausreißern."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cols = NUMERIC_COLS
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(y=df[col], ax=ax, color="#A23B72")
        ax.set_title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_boxplots_ausreisser.png", dpi=150)
    plt.close()
    print(f"📊 Gespeichert: 02_boxplots_ausreisser.png")


def plot_price_by_bhk(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    order = sorted(df["bhk"].unique())
    sns.boxplot(data=df, x="bhk", y="price", order=order, color="#F18F01")
    plt.title("Preis (Lakhs) nach Anzahl der Zimmer (BHK)")
    plt.xlabel("BHK (Zimmeranzahl)")
    plt.ylabel("Preis (Lakhs INR)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_preis_nach_bhk.png", dpi=150)
    plt.close()
    print(f"📊 Gespeichert: 03_preis_nach_bhk.png")


def plot_top_locations(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    top_locations = (
        df[df["location_grouped"] != "other"]
        .groupby("location_grouped")["price_per_sqft"]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    sns.barplot(x=top_locations.values, y=top_locations.index, color="#3B8686")
    plt.title("Top 15 Locations nach mittlerem Preis pro sqft")
    plt.xlabel("Median Preis pro sqft (INR)")
    plt.ylabel("Location")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_top_locations_preis.png", dpi=150)
    plt.close()
    print(f"📊 Gespeichert: 04_top_locations_preis.png")


def plot_area_type_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    df["area_type"].value_counts().plot(
        kind="bar", color=["#2E86AB", "#A23B72", "#F18F01", "#3B8686"]
    )
    plt.title("Verteilung area_type")
    plt.xlabel("Area Type")
    plt.ylabel("Anzahl")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_area_type_verteilung.png", dpi=150)
    plt.close()
    print(f"📊 Gespeichert: 05_area_type_verteilung.png")


def main():
    print("=== Schritt 3: Deskriptive Statistik und Datenverständnis ===\n")
    df = load_clean_data()

    stats = lageparameter_und_streuung(df)
    print("\n--- Lageparameter und Streuungsmaße ---")
    print(stats.round(2))
    stats.round(3).to_csv(f"{OUTPUT_DIR}/descriptive_stats.csv")
    print(f"\n✅ Gespeichert: descriptive_stats.csv")

    datenqualitaet_report(df)

    plot_distributions(df)
    plot_boxplots(df)
    plot_price_by_bhk(df)
    plot_top_locations(df)
    plot_area_type_distribution(df)

    print("\n--- Interpretation (Kurzfassung) ---")
    print(f"""
- Die Zielvariable 'price' ist rechtsschief verteilt (Skewness = {df['price'].skew():.2f}):
  viele günstige Immobilien, wenige sehr teure Ausreißer nach oben.
- 'total_sqft_clean' und 'price' korrelieren stark positiv (siehe Schritt 4).
- BHK-Werte > 6 sind selten und könnten in der Modellierung gesondert betrachtet werden.
- Nach der Bereinigung sind keine fehlenden Werte mehr vorhanden.
- Die Preisstreuung unterscheidet sich deutlich zwischen den Locations - ein Hinweis
  darauf, dass 'location' ein wichtiger Einflussfaktor für 'price' ist.
""")


if __name__ == "__main__":
    main()
