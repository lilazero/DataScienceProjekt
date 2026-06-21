"""
05_zusammenhangsanalyse.py

Aufgabenbereich: Zusammenhangsanalyse

  - Auswahl geeigneter Zusammenhangsmaße
  - Untersuchung der Beziehungen zwischen Variablen
  - Bestimmung von Stärke und Richtung der Zusammenhänge
  - Auswahl geeigneter Verfahren
  - Interpretation der Ergebnisse

Verwendete Verfahren:
  - Pearson-Korrelation (lineare Zusammenhänge zwischen metrischen Variablen)
  - Spearman-Korrelation (monotone, auch nicht-lineare Zusammenhänge,
    robuster gegenüber Ausreißern)
  - Cramér's V (Zusammenhang zwischen zwei kategorialen Variablen)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from config import OUTPUT_DIR

sns.set_theme(style="whitegrid")
CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"

NUMERIC_COLS = ["total_sqft_clean", "bhk", "bath", "balcony", "price", "price_per_sqft"]


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV_PATH)


def korrelationsmatrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """method: 'pearson' oder 'spearman'"""
    return df[NUMERIC_COLS].corr(method=method)


def plot_heatmap(corr: pd.DataFrame, title: str, filename: str) -> None:
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"label": "Korrelationskoeffizient"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"📊 Gespeichert: {filename}")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Cramér's V: Zusammenhangsmaß für zwei kategoriale Variablen,
    normiert auf den Bereich [0, 1] (0 = kein Zusammenhang, 1 = perfekter
    Zusammenhang). Basiert auf dem Chi-Quadrat-Test.
    """
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    if min_dim == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min_dim))


def staerke_einordnen(r: float) -> str:
    """Faustregel zur verbalen Einordnung der Korrelationsstärke (nach Cohen)."""
    r_abs = abs(r)
    if r_abs < 0.1:
        return "sehr schwach / kein Zusammenhang"
    elif r_abs < 0.3:
        return "schwach"
    elif r_abs < 0.5:
        return "mittel"
    elif r_abs < 0.7:
        return "stark"
    else:
        return "sehr stark"


def main():
    print("=== Schritt 5: Zusammenhangsanalyse ===\n")
    df = load_clean_data()

    # --- Pearson (lineare Zusammenhänge) -----------------------------
    pearson_corr = korrelationsmatrix(df, "pearson")
    print("--- Pearson-Korrelationsmatrix ---")
    print(pearson_corr.round(3))
    pearson_corr.round(3).to_csv(f"{OUTPUT_DIR}/korrelation_pearson.csv")
    plot_heatmap(pearson_corr, "Pearson-Korrelation (lineare Zusammenhänge)",
                 "08_korrelation_pearson.png")

    # --- Spearman (monotone Zusammenhänge, robuster) -------------------
    spearman_corr = korrelationsmatrix(df, "spearman")
    print("\n--- Spearman-Korrelationsmatrix ---")
    print(spearman_corr.round(3))
    spearman_corr.round(3).to_csv(f"{OUTPUT_DIR}/korrelation_spearman.csv")
    plot_heatmap(spearman_corr, "Spearman-Korrelation (monotone Zusammenhänge)",
                 "09_korrelation_spearman.png")

    # --- Stärkste Zusammenhänge mit der Zielvariable price -------------
    price_corr = pearson_corr["price"].drop("price").sort_values(key=abs, ascending=False)
    print("\n--- Stärke und Richtung der Zusammenhänge mit 'price' (Pearson) ---")
    for var, r in price_corr.items():
        richtung = "positiv" if r > 0 else "negativ"
        print(f"  {var:20s} r = {r:+.3f}  ->  {staerke_einordnen(r)}, {richtung}")

    # --- Kategoriale Zusammenhänge: Cramér's V --------------------------
    print("\n--- Cramér's V: Zusammenhang zwischen kategorialen Variablen ---")
    cv = cramers_v(df["location_grouped"], df["area_type"])
    print(f"  location_grouped <-> area_type: Cramér's V = {cv:.3f} ({staerke_einordnen(cv)})")

    cv_bhk_area = cramers_v(df["bhk"].astype(str), df["area_type"])
    print(f"  bhk <-> area_type: Cramér's V = {cv_bhk_area:.3f} ({staerke_einordnen(cv_bhk_area)})")

    print(f"""
--- Interpretation ---
- 'total_sqft_clean' zeigt mit r = {pearson_corr.loc['total_sqft_clean','price']:.2f} den stärksten
  linearen Zusammenhang mit 'price' - größere Immobilien sind im Mittel teurer.
- Pearson- und Spearman-Korrelation liefern für die meisten Variablen ähnliche
  Werte, was auf einen überwiegend linearen/monotonen Zusammenhang hindeutet.
- 'bath' und 'bhk' korrelieren stark miteinander ({pearson_corr.loc['bath','bhk']:.2f}) - 
  ein Hinweis auf Multikollinearität, die bei der Regression beachtet werden sollte.
- Die Zusammenhänge sind insgesamt plausibel und stimmen mit der Erwartung
  aus der Immobilienökonomie überein (mehr Fläche/Zimmer -> höherer Preis).
""")


if __name__ == "__main__":
    main()
