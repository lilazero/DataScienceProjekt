# Webinterface – Schnellstart

## Installation

```bash
cd project
pip install -r requirements.txt
```

(Streamlit wird bereits in `requirements.txt` eingeschlossen)

## Starten der Webapp

```bash
cd src
streamlit run 10_web_app.py
```

Die App öffnet sich automatisch im Browser unter:
**http://localhost:8501**

## Verwendung

1. **Eingabeformular (linke Seite):**
   - Wählen Sie eine **Lage** aus der Liste
   - Schieber für **Wohnfläche** (sqft)
   - Wählen Sie **BHK** (Zimmeranzahl)
   - Schieber für **Badezimmer** und **Balkone**
   - Wählen Sie die **Flächenart**

2. **Vorhersage:**
   - Klicken Sie auf den Button **"🔍 Preis schätzen"**

3. **Ergebnisse (rechte Seite):**
   - **Geschätzter Kaufpreis in USD** 💵 (mit Lakhs INR als Referenz)
   - Preiskategorie (günstig/mittel/teuer)
   - Modell-Vertrauen (Wahrscheinlichkeiten pro Kategorie)
   - Vergleich mit der Durchschnittslage (in USD)

## Beispiel

- **Location:** Whitefield
- **Wohnfläche:** 1500 sqft
- **BHK:** 3 Zimmer
- **Badezimmer:** 2
- **Balkone:** 1
- **Flächenart:** Super built-up Area

→ Das Modell schätzt z.B.: **$120,482 USD** (100 Lakhs INR)

## Währungskonvertierung

- **Basis:** Preise sind ursprünglich in **Lakhs INR** (1 Lakh = 100,000 INR)
- **Wechselkurs:** 1 USD = **83 INR** (aktuell, Januar 2025)
- **Formel:** Price_USD = (Price_Lakhs × 100,000) / 83

Um den Wechselkurs anzupassen (z.B. wenn sich der Kurs ändert):

Öffnen Sie `src/10_web_app.py` und ändern Sie:
```python
EXCHANGE_RATE = 83.0  # Hier den Kurs ändern (z.B. 85.0 für 1 USD = 85 INR)
```

## Technische Details

- **Framework:** Streamlit
- **Regression Model:** Gradient Boosting (R² ≈ 0.61)
- **Classifier:** Logistic Regression (Accuracy ≈ 76%)
- **Data:** Bengaluru House Price Dataset (11.246 Properties)
- **Currency:** USD (converted from Lakhs INR at 1 USD = 83 INR)

## Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'streamlit'"
- **Lösung:** `pip install streamlit --break-system-packages`

**Problem:** "FileNotFoundError: best_model.joblib"
- **Lösung:** Stellen Sie sicher, dass Sie aus dem `src/`-Ordner starten UND dass die Modelle in `../output/` vorhanden sind (sollten nach dem Ausführen der Skripte dort sein).

**Problem:** Die App startet aber zeigt keine Ergebnisse
- **Lösung:** Füllen Sie das Formular komplett aus und klicken Sie auf "Preis schätzen"

## Performance

Die App lädt die Modelle beim Start einmal in den Cache, daher ist die erste 
Vorhersage schnell und weitere sind sofort (< 100ms).
