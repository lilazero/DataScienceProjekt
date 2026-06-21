"""
08_klassifikation.py

Aufgabenbereich: Machine Learning und Vorhersagemodelle
"Implementierung, Vergleich und Interpretation von mindestens zwei
Klassifikationsmethoden (optional, Bonusbewertung)"

Da der Datensatz keine natürliche Klassenvariable enthält, wird die
Zielvariable 'price' in drei Preiskategorien eingeteilt:
  - cheap      (günstig)
  - medium     (mittel)
  - expensive  (teuer)

Die Grenzen werden anhand der Terzile (33%- und 67%-Quantil) der
Preisverteilung bestimmt, sodass alle drei Klassen ungefähr gleich groß sind
(wichtig für aussagekräftige Klassifikationsmetriken).

Verglichene Modelle:
  1. Logistische Regression (lineares, gut interpretierbares Baseline-Modell)
  2. Decision Tree Classifier (nicht-lineares, ebenfalls gut interpretierbares
     Modell - die Entscheidungsregeln lassen sich direkt nachvollziehen)

Bewertungsmetriken: Accuracy, Precision, Recall, F1-Score (je Klasse und
gemittelt), Konfusionsmatrix.
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

from config import OUTPUT_DIR

sns.set_theme(style="whitegrid")
CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"
CLASSIFIER_PATH = f"{OUTPUT_DIR}/best_classifier.joblib"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["total_sqft_clean", "bhk", "bath", "balcony"]
CATEGORICAL_FEATURES = ["location_grouped", "area_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLASS_LABELS = ["cheap", "medium", "expensive"]


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV_PATH)


def create_price_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt die kategoriale Zielvariable 'price_category' aus 'price'
    mittels Terzil-Einteilung (gleich große Gruppen).
    """
    df = df.copy()
    q1, q2 = df["price"].quantile([1 / 3, 2 / 3])
    df["price_category"] = pd.cut(
        df["price"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=CLASS_LABELS,
    )
    print(f"Klassengrenzen: cheap <= {q1:.1f} Lakhs < medium <= {q2:.1f} Lakhs < expensive")
    print(f"\nKlassenverteilung:\n{df['price_category'].value_counts()}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Gleiches Vorgehen wie bei der Regression: Robust Scaling + One-Hot-Encoding."""
    return ColumnTransformer(transformers=[
        ("num", RobustScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_classifiers() -> dict:
    return {
        "Logistische Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE,
        ),
    }


def plot_confusion_matrices(results: dict) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            cbar=False,
        )
        ax.set_title(f"{name}\nAccuracy = {res['accuracy']:.3f}")
        ax.set_xlabel("Vorhergesagte Klasse")
        ax.set_ylabel("Tatsächliche Klasse")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/13_klassifikation_confusion_matrices.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 13_klassifikation_confusion_matrices.png")


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    plot_df = comparison_df.melt(
        id_vars="Modell", value_vars=metrics, var_name="Metrik", value_name="Wert"
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metrik", y="Wert", hue="Modell")
    plt.title("Klassifikation: Modellvergleich (Testset)")
    plt.ylim(0, 1)
    plt.ylabel("Wert")
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/14_klassifikation_modellvergleich.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 14_klassifikation_modellvergleich.png")


def plot_decision_tree(pipeline: Pipeline, preprocessor: ColumnTransformer) -> None:
    """Visualisiert die obersten Ebenen des Decision Tree zur Interpretation."""
    tree_model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model, max_depth=3, feature_names=feature_names,
        class_names=CLASS_LABELS, filled=True, fontsize=9, proportion=True,
    )
    plt.title("Decision Tree zur Preiskategorisierung (oberste 3 Ebenen)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/15_decision_tree_visualisierung.png", dpi=150)
    plt.close()
    print("📊 Gespeichert: 15_decision_tree_visualisierung.png")


def main():
    print("=== Schritt 8: Klassifikation (Bonus) ===\n")
    df = load_clean_data()
    df = create_price_category(df)

    X = df[ALL_FEATURES]
    y = df["price_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\nTrainingsdaten: {len(X_train)} Zeilen, Testdaten: {len(X_test)} Zeilen "
          f"(80/20-Split, stratifiziert nach Klasse)")

    preprocessor = build_preprocessor()
    classifiers = get_classifiers()
    results = {}
    comparison_rows = []
    fitted_pipelines = {}

    for name, clf in classifiers.items():
        print(f"\n--- Training: {name} ---")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_acc = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="accuracy")
        print(f"  CV Accuracy: {cv_acc.mean():.4f} (± {cv_acc.std():.4f})")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", labels=CLASS_LABELS,
        )
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

        print(f"  Test Accuracy:  {accuracy:.4f}")
        print(f"  Test Precision (macro): {precision:.4f}")
        print(f"  Test Recall (macro):    {recall:.4f}")
        print(f"  Test F1-Score (macro):  {f1:.4f}")
        print(f"\n  Klassifikationsbericht:\n{classification_report(y_test, y_pred, labels=CLASS_LABELS)}")

        results[name] = {"accuracy": accuracy, "confusion_matrix": cm, "y_pred": y_pred}
        comparison_rows.append({
            "Modell": name, "accuracy": accuracy,
            "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
            "cv_accuracy_mean": cv_acc.mean(), "cv_accuracy_std": cv_acc.std(),
        })
        fitted_pipelines[name] = pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values("accuracy", ascending=False)
    print("\n--- Modellvergleich (sortiert nach Accuracy) ---")
    print(comparison_df.round(4).to_string(index=False))
    comparison_df.round(4).to_csv(f"{OUTPUT_DIR}/klassifikation_modellvergleich.csv", index=False)

    plot_confusion_matrices(results)
    plot_model_comparison(comparison_df)

    best_name = comparison_df.iloc[0]["Modell"]
    best_pipeline = fitted_pipelines[best_name]
    joblib.dump(best_pipeline, CLASSIFIER_PATH)
    print(f"\n✅ Bestes Klassifikationsmodell ('{best_name}') gespeichert unter: {CLASSIFIER_PATH}")

    if "Decision Tree" in fitted_pipelines:
        plot_decision_tree(fitted_pipelines["Decision Tree"], preprocessor)

        importances = pd.DataFrame({
            "feature": preprocessor.get_feature_names_out(),
            "importance": fitted_pipelines["Decision Tree"].named_steps["model"].feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"\nWichtigste Splits im Decision Tree:\n{importances.head(8).to_string(index=False)}")

    print(f"""
--- Interpretation ---
- Beide Modelle erreichen eine Accuracy von über {comparison_df['accuracy'].min()*100:.0f}%
  bei 3 etwa gleich großen Klassen (Zufallsniveau wäre ca. 33%).
- '{best_name}' erzielt die höhere Accuracy ({comparison_df.iloc[0]['accuracy']*100:.1f}%).
- Logistische Regression liefert ein lineares, leicht interpretierbares Modell
  (Koeffizienten zeigen Richtung & Stärke des Einflusses je Klasse).
- Der Decision Tree bildet nicht-lineare Entscheidungsregeln ab und ist durch
  die Baumstruktur direkt nachvollziehbar (siehe 15_decision_tree_visualisierung.png).
- Fehlklassifikationen treten erwartungsgemäß am häufigsten zwischen benachbarten
  Klassen auf (cheap<->medium, medium<->expensive), siehe Konfusionsmatrix -
  die Modelle verwechseln selten 'cheap' direkt mit 'expensive'.
""")


if __name__ == "__main__":
    main()
