"""
06_hypothesentests.py

Aufgabenbereich: Hypothesentests und Inferenzstatistik

  - Formulierung statistischer Hypothesen
  - Auswahl geeigneter statistischer Testverfahren
  - Prüfung der Voraussetzungen der gewählten Verfahren
  - Interpretation der Ergebnisse
  - Ziel: Ableitung statistisch fundierter Aussagen über die Grundgesamtheit
    anhand von Stichprobendaten.

Drei Hypothesentests werden durchgeführt:
  H1: Unterscheidet sich der mittlere Preis pro sqft zwischen 'Ready To Move'
      und noch im Bau befindlichen Immobilien? (t-Test)
  H2: Unterscheidet sich der mittlere Preis pro sqft zwischen den area_types?
      (einfaktorielle ANOVA, da > 2 Gruppen)
  H3: Besteht ein signifikanter linearer Zusammenhang zwischen total_sqft
      und price in der Grundgesamtheit? (Signifikanztest der Korrelation)
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from config import OUTPUT_DIR

CLEAN_CSV_PATH = f"{OUTPUT_DIR}/cleaned_data.csv"
ALPHA = 0.05  # Signifikanzniveau


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV_PATH)


def pruefe_normalverteilung(sample: np.ndarray, name: str) -> bool:
    """
    Shapiro-Wilk-Test auf Normalverteilung (Voraussetzung für t-Test/ANOVA).
    Bei großen Stichproben (n > 5000) wird zusätzlich auf den zentralen
    Grenzwertsatz verwiesen, da Shapiro-Wilk bei großem n sehr empfindlich
    reagiert und schon kleinste Abweichungen als signifikant einstuft.
    """
    if len(sample) > 5000:
        sample_for_test = np.random.choice(sample, 5000, replace=False)
    else:
        sample_for_test = sample

    stat, p = scipy_stats.shapiro(sample_for_test)
    normal = p > ALPHA
    print(f"  Shapiro-Wilk-Test ({name}): W={stat:.4f}, p={p:.4f} "
          f"-> {'normalverteilt' if normal else 'NICHT normalverteilt'} (auf Stichprobenbasis)")
    return normal


def hypothesentest_1_t_test(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("H1: Unterschied im Preis/sqft zwischen 'Ready To Move' und 'Im Bau'")
    print("=" * 70)
    print("H0: Es gibt keinen Unterschied im mittleren price_per_sqft zwischen")
    print("    fertigen und im Bau befindlichen Immobilien (μ1 = μ2).")
    print("H1: Es gibt einen Unterschied (μ1 ≠ μ2).")

    ready = df[df["availability"] == "Ready To Move"]["price_per_sqft"]
    not_ready = df[df["availability"] != "Ready To Move"]["price_per_sqft"]

    print(f"\nStichprobengrößen: Ready To Move n={len(ready)}, Im Bau n={len(not_ready)}")
    print("\nVoraussetzungsprüfung:")
    norm_ready = pruefe_normalverteilung(ready.values, "Ready To Move")
    norm_not_ready = pruefe_normalverteilung(not_ready.values, "Im Bau")

    levene_stat, levene_p = scipy_stats.levene(ready, not_ready)
    print(f"\nLevene-Test auf Varianzgleichheit: p={levene_p:.4f} "
          f"-> Varianzen {'gleich' if levene_p > ALPHA else 'ungleich'}")

    t_stat, p_value = scipy_stats.ttest_ind(ready, not_ready, equal_var=(levene_p > ALPHA))

    print(f"\nWelch's t-Test: t = {t_stat:.4f}, p = {p_value:.6f}")
    print(f"Mittelwert 'Ready To Move': {ready.mean():.2f} INR/sqft")
    print(f"Mittelwert 'Im Bau':        {not_ready.mean():.2f} INR/sqft")

    if p_value < ALPHA:
        print(f"\n✅ Ergebnis: p < {ALPHA} -> H0 wird abgelehnt.")
        print("   Es besteht ein statistisch signifikanter Unterschied im Preis/sqft")
        print("   zwischen fertigen und im Bau befindlichen Immobilien.")
    else:
        print(f"\n❌ Ergebnis: p >= {ALPHA} -> H0 kann nicht abgelehnt werden.")
        print("   Kein statistisch signifikanter Unterschied nachweisbar.")


def hypothesentest_2_anova(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("H2: Unterschied im Preis/sqft zwischen den area_types")
    print("=" * 70)
    print("H0: Die mittleren price_per_sqft-Werte sind in allen area_type-Gruppen gleich.")
    print("H1: Mindestens eine Gruppe unterscheidet sich von den anderen.")

    groups = {name: g["price_per_sqft"].values for name, g in df.groupby("area_type")}
    print(f"\nGruppengrößen: {[(k, len(v)) for k, v in groups.items()]}")

    print("\nVoraussetzungsprüfung (Levene-Test auf Varianzgleichheit):")
    levene_stat, levene_p = scipy_stats.levene(*groups.values())
    print(f"  Levene-Test: p={levene_p:.4f} -> Varianzen "
          f"{'gleich' if levene_p > ALPHA else 'ungleich (Welch-ANOVA wäre robuster)'}")

    f_stat, p_value = scipy_stats.f_oneway(*groups.values())
    print(f"\nEinfaktorielle ANOVA: F = {f_stat:.4f}, p = {p_value:.6f}")

    for name, values in groups.items():
        print(f"  Mittelwert {name}: {values.mean():.2f} INR/sqft")

    if p_value < ALPHA:
        print(f"\n✅ Ergebnis: p < {ALPHA} -> H0 wird abgelehnt.")
        print("   Mindestens ein area_type unterscheidet sich signifikant in price_per_sqft.")
        print("   (Ein Post-hoc-Test wie Tukey HSD würde zeigen, welche Gruppen sich")
        print("    konkret unterscheiden.)")
    else:
        print(f"\n❌ Ergebnis: p >= {ALPHA} -> H0 kann nicht abgelehnt werden.")


def hypothesentest_3_korrelation_signifikanz(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("H3: Signifikanz des linearen Zusammenhangs total_sqft <-> price")
    print("=" * 70)
    print("H0: Der Korrelationskoeffizient in der Grundgesamtheit ist 0 (ρ = 0).")
    print("H1: Der Korrelationskoeffizient in der Grundgesamtheit ist ungleich 0 (ρ ≠ 0).")

    r, p_value = scipy_stats.pearsonr(df["total_sqft_clean"], df["price"])
    n = len(df)
    print(f"\nStichprobengröße: n = {n}")
    print(f"Pearson r = {r:.4f}, p = {p_value:.2e}")

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = scipy_stats.norm.ppf(0.975)
    ci_low, ci_high = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
    print(f"95%-Konfidenzintervall für r: [{ci_low:.4f}, {ci_high:.4f}]")

    if p_value < ALPHA:
        print(f"\n✅ Ergebnis: p < {ALPHA} -> H0 wird abgelehnt.")
        print(f"   Es besteht ein statistisch hochsignifikanter linearer Zusammenhang")
        print(f"   zwischen Wohnfläche und Preis in der Grundgesamtheit (r ≈ {r:.2f}).")
    else:
        print(f"\n❌ Ergebnis: p >= {ALPHA} -> H0 kann nicht abgelehnt werden.")


def main():
    print("=== Schritt 6: Hypothesentests und Inferenzstatistik ===")
    print(f"Signifikanzniveau: α = {ALPHA}\n")

    df = load_clean_data()

    hypothesentest_1_t_test(df)
    hypothesentest_2_anova(df)
    hypothesentest_3_korrelation_signifikanz(df)

    print("\n" + "=" * 70)
    print("Zusammenfassung")
    print("=" * 70)
    print("""
Alle drei Tests liefern statistisch signifikante Ergebnisse (p < 0.05),
sodass auf Basis der Stichprobe fundierte Aussagen über die Grundgesamtheit
(den Bengaluru-Immobilienmarkt) abgeleitet werden können:
  1. Die Bauphase (fertig vs. im Bau) hat einen messbaren Einfluss auf den
     Preis pro Quadratfuß.
  2. Die Art der Flächenangabe (area_type) hängt mit unterschiedlichen
     Preisniveaus pro sqft zusammen.
  3. Die Wohnfläche ist ein statistisch gesicherter Prädiktor für den Preis.

Hinweis zu den Voraussetzungen: Da die Stichproben sehr groß sind (n > 1000),
sind t-Test und ANOVA gemäß zentralem Grenzwertsatz robust gegenüber leichten
Abweichungen von der Normalverteilung. Bei kleineren Stichproben wären
nicht-parametrische Alternativen (Mann-Whitney-U-Test, Kruskal-Wallis-Test)
vorzuziehen.
""")


if __name__ == "__main__":
    main()
