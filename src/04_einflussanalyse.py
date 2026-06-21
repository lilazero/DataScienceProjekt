"""
04_einflussanalyse.py

Aufgabenbereich: Einflussanalyse

  - Definition der Zielvariable und Einflussvariablen
  - Analyse des Einflusses einzelner Variablen auf die Zielvariable
  - Analyse der gemeinsamen Wirkung der Einflussvariablen
  - Bestimmung der wichtigsten Einflussfaktoren
  - Interpretation und Vergleich der Ergebnisse
  - Ableitung von Schlussfolgerungen für die Problemstellung

Zielvariable: price
Mögliche Einflussvariablen: total_sqft_clean, bhk, bath, balcony,
                             location_grouped, area_type
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from config import OUTPUT_DIR, TARGET_VARIABLE

sns.set_theme(style="whitegrid")
CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"

NUMERIC_FEATURES = ["total_sqft_clean", "bhk", "bath", "balcony"]
CATEGORICAL_FEATURES = ["location_grouped", "area_type"]


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV_PATH)


def einzeleinfluss_numerisch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysiert den Einfluss einzelner numerischer Variablen auf 'price'
    mittels Pearson- und Spearman-Korrelation.
    """
    results = []
    for col in NUMERIC_FEATURES:
        pearson_r, pearson_p = scipy_stats.pearsonr(df[col], df[TARGET_VARIABLE])
        spearman_r, spearman_p = scipy_stats.spearmanr(df[col], df[TARGET_VARIABLE])
        results.append({
            "variable": col,
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p_value": spearman_p,
        })
    return pd.DataFrame(results).sort_values("pearson_r", ascending=False)


def einzeleinfluss_kategorisch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysiert den Einfluss kategorialer Variablen auf 'price' mittels
    einfaktorieller ANOVA (Unterscheiden sich die Gruppenmittelwerte signifikant?).
    """
    results = []
    for col in CATEGORICAL_FEATURES:
        groups = [g[TARGET_VARIABLE].values for _, g in df.groupby(col) if len(g) >= 5]
        f_stat, p_value = scipy_stats.f_oneway(*groups)
        results.append({
            "variable": col,
            "f_statistic": f_stat,
            "p_value": p_value,
            "n_groups": len(groups),
            "signifikant (p<0.05)": p_value < 0.05,
        })
    return pd.DataFrame(results)


def plot_scatter_with_trend(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        sns.regplot(
            data=df, x=col, y=TARGET_VARIABLE, ax=ax,
            scatter_kws={"alpha": 0.3, "s": 10, "color": "#2E86AB"},
            line_kws={"color": "#A23B72"},
        )
        ax.set_title(f"{col} vs. {TARGET_VARIABLE}")
        if col == "total_sqft_clean":
            ax.set_xlim(0, df[col].quantile(0.99))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_einzeleinfluss_scatterplots.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 06_einzeleinfluss_scatterplots.png")


def gemeinsame_wirkung_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysiert die GEMEINSAME Wirkung aller Einflussvariablen mittels
    Random-Forest-Feature-Importance. Dies berücksichtigt im Gegensatz zur
    Einzelkorrelation auch Wechselwirkungen zwischen den Variablen.
    """
    df_model = df.copy()

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_model[col + "_encoded"] = le.fit_transform(df_model[col])
        encoders[col] = le

    feature_cols = NUMERIC_FEATURES + [c + "_encoded" for c in CATEGORICAL_FEATURES]
    X = df_model[feature_cols]
    y = df_model[TARGET_VARIABLE]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance


def plot_feature_importance(importance: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#3B8686")
    plt.title("Gemeinsame Wirkung der Einflussvariablen\n(Random-Forest Feature Importance)")
    plt.xlabel("Relative Wichtigkeit")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 07_feature_importance.png")


def main():
    print("=== Schritt 4: Einflussanalyse ===\n")
    df = load_clean_data()

    print(f"Zielvariable: {TARGET_VARIABLE}")
    print(f"Numerische Einflussvariablen: {NUMERIC_FEATURES}")
    print(f"Kategoriale Einflussvariablen: {CATEGORICAL_FEATURES}\n")

    print("--- Einzeleinfluss: numerische Variablen (Korrelation mit price) ---")
    num_results = einzeleinfluss_numerisch(df)
    print(num_results.round(4))
    num_results.to_csv(f"{OUTPUT_DIR}/einfluss_numerisch.csv", index=False)

    print("\n--- Einzeleinfluss: kategoriale Variablen (ANOVA) ---")
    cat_results = einzeleinfluss_kategorisch(df)
    print(cat_results.round(4))
    cat_results.to_csv(f"{OUTPUT_DIR}/einfluss_kategorisch.csv", index=False)

    plot_scatter_with_trend(df)

    print("\n--- Gemeinsame Wirkung aller Variablen (Random Forest Importance) ---")
    importance = gemeinsame_wirkung_feature_importance(df)
    print(importance.round(4))
    importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    plot_feature_importance(importance)

    top_feature = importance.iloc[0]["feature"]
    print(f"""
--- Interpretation ---
- Einzelbetrachtung: '{num_results.iloc[0]['variable']}' zeigt die stärkste
  lineare Korrelation mit 'price' (Pearson r = {num_results.iloc[0]['pearson_r']:.3f}).
- Bei gemeinsamer Betrachtung aller Variablen (inkl. Wechselwirkungen) ist
  '{top_feature}' der wichtigste Einflussfaktor.
- Sowohl numerische (Fläche, Zimmerzahl) als auch kategoriale Faktoren
  (Location) tragen signifikant zur Preisbildung bei.
- Schlussfolgerung: Ein realistisches Vorhersagemodell für 'price' muss
  sowohl die Größe der Immobilie als auch die Lage berücksichtigen.
""")


if __name__ == "__main__":
    main()
