"""
10_web_app.py

Streamlit Web-App für die Immobilienpreis-Schätzung

Funktionalität:
  - Interaktives Input-Formular mit allen 6 Features
  - Vorhersage des Preises (Regression) mittels Gradient Boosting
  - Vorhersage der Preiskategorie (Klassifikation) mittels Logistischer Regression
  - Kontext-Informationen zur ausgewählten Lage
  - Validierung der Eingaben

Aufruf:
    streamlit run 10_web_app.py

Die App öffnet sich automatisch im Browser unter http://localhost:8501
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ============================================================================
# Konfiguration
# ============================================================================

st.set_page_config(
    page_title="Bengaluru Immobilienpreis-Schätzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Wechselkurs: 1 USD = X INR
# (Sie können diesen Wert anpassen, falls der Kurs sich ändert)
EXCHANGE_RATE = 83.0  # 1 USD = 83 INR (aktueller Kurs, Januar 2025)

# Pfade zu Modellen und Daten (relativ zum Skript-Verzeichnis)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "output")

REGRESSION_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
CLASSIFIER_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_classifier.joblib")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_data.csv")

# ============================================================================
# Cache-Loading (damit Modelle nur einmal geladen werden)
# ============================================================================

def lakhs_inr_to_usd(price_lakhs: float, exchange_rate: float = EXCHANGE_RATE) -> float:
    """
    Konvertiert Preis von Lakhs INR zu USD.
    
    1 Lakh = 100,000 INR
    price_usd = (price_lakhs * 100,000) / exchange_rate
    """
    price_inr = price_lakhs * 100_000
    price_usd = price_inr / exchange_rate
    return price_usd

@st.cache_resource
def load_models_and_data():
    """Laden Sie Modelle und Daten einmal beim App-Start."""
    try:
        regression_model = joblib.load(REGRESSION_MODEL_PATH)
        classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
        cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
        return regression_model, classifier_model, cleaned_data
    except FileNotFoundError as e:
        st.error(f"❌ Fehler beim Laden der Modelle: {e}")
        st.stop()


regression_model, classifier_model, df_clean = load_models_and_data()

# ============================================================================
# Daten vorbereiten
# ============================================================================

# Eindeutige Locations (sortiert)
locations = sorted(df_clean['location_grouped'].unique())

# Area-Types (aus den Daten)
area_types = sorted(df_clean['area_type'].unique())

# Statistiken pro Location (für Kontext)
location_stats = df_clean.groupby('location_grouped')['price'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)

# Preis-Kategorisierungssgrenzen (aus dem Klassifikations-Skript)
PRICE_CHEAP_LIMIT = 55.0
PRICE_MEDIUM_LIMIT = 90.0
CLASS_LABELS = ['cheap', 'medium', 'expensive']
CLASS_DESCRIPTIONS = {
    'cheap': '💰 Günstig',
    'medium': '💵 Mittel',
    'expensive': '💎 Teuer'
}

# ============================================================================
# Seitenlayout & Titel
# ============================================================================

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 Bengaluru Immobilienpreis-Schätzer (USD)")
st.markdown("""
Geben Sie die Eckdaten einer Immobilie an, und unser Machine-Learning-Modell
schätzt automatisch:
- **Kaufpreis in USD** (Regression: Gradient Boosting Modell, R² ≈ 0.61)
- **Preiskategorie** (Klassifikation: Logistische Regression, Accuracy ≈ 76%)

*Basierend auf dem Bengaluru House Price Dataset (Kaggle)*  
*Preise werden von Lakhs INR mit aktuellem Kurs (1 USD = {EXCHANGE_RATE} INR) in USD konvertiert*
""")

st.divider()

# ============================================================================
# Input-Formular (links) + Ergebnisse (rechts)
# ============================================================================

col_input, col_results = st.columns([1, 1], gap="large")

# --- INPUT COLUMN ---
with col_input:
    st.subheader("📝 Immobilien-Details")
    
    # Lage
    location = st.selectbox(
        "**Lage**",
        options=locations,
        index=locations.index('other') if 'other' in locations else 0,
        help="Wählen Sie die Location der Immobilie"
    )
    
    # Wohnfläche
    sqft = st.slider(
        "**Wohnfläche (sqft)**",
        min_value=300,
        max_value=5000,
        value=1500,
        step=50,
        help="Gesamte Wohnfläche in Quadratfuß (1 sqft ≈ 0.09 m²)"
    )
    
    # BHK (Zimmeranzahl)
    bhk = st.selectbox(
        "**BHK (Zimmeranzahl)**",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=1,  # Default: 2 BHK
        help="Bedroom-Hall-Kitchen: Anzahl der Wohnzimmer"
    )
    
    # Badezimmer
    bath = st.slider(
        "**Badezimmer**",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Anzahl der Badezimmer"
    )
    
    # Balkone
    balcony = st.slider(
        "**Balkone**",
        min_value=0,
        max_value=3,
        value=1,
        step=1,
        help="Anzahl der Balkone/Terrassen"
    )
    
    # Area Type
    area_type = st.selectbox(
        "**Flächenart**",
        options=area_types,
        index=0,
        help="Art der Flächenangabe (Super built-up, Built-up, oder Plot)"
    )
    
    # Vorhersage-Button
    st.divider()
    predict_button = st.button(
        "🔍 Preis schätzen",
        use_container_width=True,
        type="primary"
    )

# --- RESULTS COLUMN ---
with col_results:
    st.subheader("📊 Vorhersage-Ergebnisse")
    
    if predict_button:
        # Input validieren
        if bath > bhk + 2:
            st.warning("⚠️ Warnung: Die Anzahl der Badezimmer ist ungewöhnlich hoch für die Zimmeranzahl.")
        
        # DataFrame für Vorhersage vorbereiten
        input_df = pd.DataFrame({
            'total_sqft_clean': [float(sqft)],
            'bhk': [int(bhk)],
            'bath': [int(bath)],
            'balcony': [int(balcony)],
            'location_grouped': [location],
            'area_type': [area_type]
        })
        
        try:
            # Regression: Preis-Vorhersage
            predicted_price = regression_model.predict(input_df)[0]
            predicted_price_usd = lakhs_inr_to_usd(predicted_price)
            
            # Klassifikation: Preiskategorie-Vorhersage
            predicted_class = classifier_model.predict(input_df)[0]
            predicted_proba = classifier_model.predict_proba(input_df)[0]
            
            # Ausgabe: Hauptergebnis (großer, prominenter Text)
            st.metric(
                label="💵 Geschätzter Kaufpreis (USD)",
                value=f"${predicted_price_usd:,.0f}",
                delta=f"{predicted_price:.1f} Lakhs INR"
            )
            
            # Ausgabe: Preiskategorie
            category_emoji = CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)
            st.success(f"**Preiskategorie:** {category_emoji}")
            
            # Ausgabe: Confidence (Wahrscheinlichkeit)
            st.divider()
            st.subheader("📈 Modell-Vertrauen")
            
            confidence_data = {
                'Kategorie': CLASS_LABELS,
                'Wahrscheinlichkeit': [f"{p*100:.1f}%" for p in predicted_proba],
                'Score': predicted_proba
            }
            confidence_df = pd.DataFrame(confidence_data)
            
            col_conf1, col_conf2 = st.columns([2, 1])
            with col_conf1:
                st.bar_chart(confidence_df.set_index('Kategorie')['Score'])
            with col_conf2:
                st.dataframe(
                    confidence_df[['Kategorie', 'Wahrscheinlichkeit']],
                    hide_index=True,
                    use_container_width=True
                )
            
            # Kontext: Vergleich mit der Lage
            st.divider()
            st.subheader("📍 Kontext: Diese Lage")
            
            if location in location_stats.index:
                stats = location_stats.loc[location]
                stats_mean_usd = lakhs_inr_to_usd(stats['mean'])
                stats_min_usd = lakhs_inr_to_usd(stats['min'])
                stats_max_usd = lakhs_inr_to_usd(stats['max'])
                
                col_c1, col_c2, col_c3 = st.columns(3)
                
                with col_c1:
                    st.metric(
                        "Immobilien in DB",
                        int(stats['count'])
                    )
                
                with col_c2:
                    st.metric(
                        "Ø Preis (USD)",
                        f"${stats_mean_usd:,.0f}",
                        f"{stats['mean']:.0f} Lakhs"
                    )
                
                with col_c3:
                    st.metric(
                        "Spanne (USD)",
                        f"${stats_min_usd:,.0f}–${stats_max_usd:,.0f}",
                        f"{stats['min']:.0f}–{stats['max']:.0f} Lakhs"
                    )
                
                # Vergleich
                price_vs_avg = ((predicted_price - stats['mean']) / stats['mean']) * 100
                if price_vs_avg > 0:
                    st.info(f"💡 Diese Immobilie liegt **{price_vs_avg:.0f}% über** dem Durchschnitt dieser Lage.")
                elif price_vs_avg < 0:
                    st.info(f"💡 Diese Immobilie liegt **{abs(price_vs_avg):.0f}% unter** dem Durchschnitt dieser Lage.")
                else:
                    st.info(f"💡 Diese Immobilie liegt **am Durchschnitt** dieser Lage.")
        
        except Exception as e:
            st.error(f"❌ Fehler bei der Vorhersage: {str(e)}")
    
    else:
        st.info("👈 Füllen Sie das Formular aus und klicken Sie auf **'Preis schätzen'**")

# ============================================================================
# Footer / Zusatz-Infos
# ============================================================================

st.divider()

with st.expander("ℹ️ Über diese App"):
    st.markdown("""
    ### Modell-Details
    
    **Regression (Preis-Vorhersage):**
    - Modell: Gradient Boosting Regressor
    - Testset R²: 0.61 (erklärt 61% der Preis-Varianz)
    - MAE: ca. 29 Lakhs INR
    
    **Klassifikation (Preiskategorie):**
    - Modelle: Logistische Regression (Gewinner) und Decision Tree
    - Testset Accuracy: 76% (Logistic Regression)
    - Klassen: cheap (≤ 55 Lakhs), medium (55–90 Lakhs), expensive (> 90 Lakhs)
    
    ### Wichtige Hinweise
    
    - Die Vorhersagen basieren auf historischen Daten aus dem Bengaluru-Immobilienmarkt
    - Die Modelle kennen keine Informationen zu Ausstattung, Baujahr oder Mikrolage
    - Diese Schätzungen sind **nicht** als Investitionsberatung zu verstehen
    - Für echte Transaktionen sollte eine professionelle Bewertung herangezogen werden
    
    ### Datenschutz
    
    - Es werden **keine persönlichen Daten** erfasst oder gespeichert
    - Diese App verarbeitet ausschließlich immobilienbezogene Merkmale
    """)

with st.expander("🛠️ Technische Details"):
    st.code("""
Framework: Streamlit
ML-Modelle: scikit-learn (Gradient Boosting, Logistic Regression, Decision Tree)
Daten: Bengaluru House Price Dataset (Kaggle)
Skalierung: RobustScaler (robust gegenüber Ausreißern)
Währungskonvertierung: Fixed Exchange Rate (1 USD = 83 INR)
    """, language="python")
    
    st.markdown(f"""
    **Aktuelle Wechselkurs-Einstellung:** 1 USD = {EXCHANGE_RATE} INR
    
    Um den Wechselkurs anzupassen, bearbeiten Sie die Zeile in `10_web_app.py`:
    ```python
    EXCHANGE_RATE = {EXCHANGE_RATE}  # Hier ändern
    ```
    """)
