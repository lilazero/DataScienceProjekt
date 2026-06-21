"""
07_regression_modelle.py

Aufgabenbereich: Machine Learning und Vorhersagemodelle (Regression - verpflichtend)

  - Entwicklung und Evaluation von Vorhersagemodellen für die Zielvariable
  - Definition relevanter Variablen und Aufbereitung der Datenbasis
  - Vorbereitung, Auswahl und Dokumentation relevanter Features
  - Training, Vergleich und Bewertung der Modelle anhand geeigneter Metriken
  - Interpretation der Ergebnisse und Diskussion der Modellgrenzen
  - Durchführung einer Regression zur Vorhersage numerischer Werte (price)

Verglichene Modelle:
  1. Lineare Regression (einfaches, interpretierbares Baseline-Modell)
  2. Ridge-Regression (lineares Modell mit L2-Regularisierung)
  3. Random Forest Regressor (nicht-lineares Ensemble-Modell)
  4. Gradient Boosting Regressor (nicht-lineares Ensemble-Modell)

Bewertungsmetriken: R², MAE, RMSE (jeweils per Cross-Validation und auf
einem unabhängigen Testset).
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import OUTPUT_DIR, TARGET_VARIABLE

sns.set_theme(style="whitegrid")
CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"
MODEL_PATH = f"{OUTPUT_DIR}/best_model.joblib"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["total_sqft_clean", "bhk", "bath", "balcony"]
CATEGORICAL_FEATURES = ["location_grouped", "area_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV_PATH)


def build_preprocessor() -> ColumnTransformer:
    """
    Erstellt die Vorverarbeitungs-Pipeline:
      - numerische Features: Robust Scaling (statt Standardisierung), da
        total_sqft_clean, bath und bhk auch nach der Bereinigung noch eine
        deutliche Rechtsschiefe und einzelne Extremwerte aufweisen
        (z.B. total_sqft_clean: 300 - 20.000, bath/bhk bis 16).
        RobustScaler verwendet Median und Interquartilsabstand (IQR) statt
        Mittelwert und Standardabweichung, wodurch der Einfluss verbleibender
        Ausreißer auf die Skalierung deutlich reduziert wird.
      - kategoriale Features: One-Hot-Encoding
    """
    return ColumnTransformer(transformers=[
        ("num", RobustScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_models() -> dict:
    return {
        "Lineare Regression": LinearRegression(),
        "Ridge-Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_with_cv(pipeline, X, y, cv_folds=5) -> dict:
    """5-fache Cross-Validation zur robusten Schätzung der Modellgüte."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    return {
        "cv_r2_mean": r2_scores.mean(), "cv_r2_std": r2_scores.std(),
        "cv_mae_mean": mae_scores.mean(),
        "cv_rmse_mean": rmse_scores.mean(),
    }


def main():
    print("=== Schritt 7: Machine Learning - Regressionsmodelle ===\n")
    df = load_clean_data()

    X = df[ALL_FEATURES]
    y = df[TARGET_VARIABLE]

    print(f"Zielvariable: {TARGET_VARIABLE} (Preis in Lakhs INR)")
    print(f"Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
    print(f"Gesamtdatensatz: {len(df)} Zeilen\n")

    # --- Train/Test-Split ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Trainingsdaten: {len(X_train)} Zeilen, Testdaten: {len(X_test)} Zeilen "
          f"(80/20-Split)\n")

    preprocessor = build_preprocessor()
    models = get_models()
    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"--- Training: {name} ---")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Cross-Validation auf den Trainingsdaten (robuste Güteschätzung)
        cv_results = evaluate_with_cv(pipeline, X_train, y_train)

        # Finales Training auf allen Trainingsdaten + Evaluation auf Testset
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Modell": name,
            "CV R² (Mittelwert)": cv_results["cv_r2_mean"],
            "CV R² (Std.abw.)": cv_results["cv_r2_std"],
            "Test R²": test_r2,
            "Test MAE (Lakhs)": test_mae,
            "Test RMSE (Lakhs)": test_rmse,
        })
        fitted_pipelines[name] = pipeline

        print(f"  CV R²:  {cv_results['cv_r2_mean']:.4f} (± {cv_results['cv_r2_std']:.4f})")
        print(f"  Test R²:   {test_r2:.4f}")
        print(f"  Test MAE:  {test_mae:.2f} Lakhs")
        print(f"  Test RMSE: {test_rmse:.2f} Lakhs\n")

    results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False)
    print("\n--- Modellvergleich (sortiert nach Test R²) ---")
    print(results_df.round(4).to_string(index=False))
    results_df.round(4).to_csv(f"{OUTPUT_DIR}/modellvergleich.csv", index=False)

    # --- Bestes Modell speichern ------------------------------------------
    best_model_name = results_df.iloc[0]["Modell"]
    best_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n✅ Bestes Modell ('{best_model_name}') gespeichert unter: {MODEL_PATH}")

    # --- Visualisierung: Modellvergleich ----------------------------------
    plt.figure(figsize=(9, 6))
    sns.barplot(data=results_df, x="Test R²", y="Modell", color="#2E86AB")
    plt.title("Modellvergleich: Test R² (Bestimmtheitsmaß)")
    plt.xlabel("R² (Anteil erklärter Varianz)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_modellvergleich_r2.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 10_modellvergleich_r2.png")

    # --- Visualisierung: Tatsächlich vs. Vorhergesagt (bestes Modell) -----
    y_pred_best = best_pipeline.predict(X_test)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_best, alpha=0.4, color="#2E86AB", s=15)
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([0, max_val], [0, max_val], color="#A23B72", linestyle="--", label="Perfekte Vorhersage")
    plt.xlabel("Tatsächlicher Preis (Lakhs)")
    plt.ylabel("Vorhergesagter Preis (Lakhs)")
    plt.title(f"Tatsächlich vs. Vorhergesagt - {best_model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_tatsaechlich_vs_vorhergesagt.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 11_tatsaechlich_vs_vorhergesagt.png")

    # --- Residuenanalyse (Diskussion der Modellgrenzen) ---------------------
    residuals = y_test.values - y_pred_best
    plt.figure(figsize=(9, 6))
    plt.scatter(y_pred_best, residuals, alpha=0.4, color="#3B8686", s=15)
    plt.axhline(0, color="#A23B72", linestyle="--")
    plt.xlabel("Vorhergesagter Preis (Lakhs)")
    plt.ylabel("Residuum (Tatsächlich - Vorhergesagt)")
    plt.title(f"Residuenanalyse - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_residuenanalyse.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 12_residuenanalyse.png")

    print(f"""
--- Interpretation und Modellgrenzen ---
- Das beste Modell ('{best_model_name}') erklärt {results_df.iloc[0]['Test R²']*100:.1f}%
  der Varianz im Immobilienpreis (Test R²).
- Der mittlere absolute Fehler (MAE) liegt bei {results_df.iloc[0]['Test MAE (Lakhs)']:.1f} Lakhs INR.
- Nicht-lineare Modelle (Random Forest, Gradient Boosting) übertreffen typischerweise
  die linearen Modelle, da der Zusammenhang zwischen Location, Fläche und Preis
  nicht rein linear ist.
- Modellgrenzen:
  * Sehr seltene Locations wurden zu 'other' zusammengefasst -> Informationsverlust
    bei selteneren, aber ggf. besonders teuren/günstigen Lagen.
  * Das Modell kennt keine Informationen zu Ausstattung, Baujahr, Infrastruktur
    oder Mikrolage (z.B. Entfernung zur Metro), die in der Praxis ebenfalls
    preisrelevant wären.
  * Bei Immobilien mit sehr hohen Preisen (Luxussegment) ist die Vorhersagegüte
    erfahrungsgemäß geringer (siehe Residuenplot: größere Streuung bei hohen Werten).
""")


if __name__ == "__main__":
    main()
