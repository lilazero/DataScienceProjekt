# Agiles Programmierprojekt – Bengaluru Immobilienpreis-Analyse

Projekt für die Weiterbildung "intoCODE / InterGeeks" an der Hochschule Hannover.

**Datensatz:** Bengaluru House Price Data (Kaggle, frei verfügbar)
**Zielvariable:** `price` (Verkaufspreis in Lakhs INR)
**Anwendungsbereich:** Immobilienmarkt Bengaluru – Analyse der Preisbildung
und Entwicklung eines Vorhersagemodells für den Immobilienpreis basierend auf
Lage, Fläche und Ausstattungsmerkmalen.

---

## Projektstruktur

```
project/
├── data/
│   └── Bengaluru_House_Data.csv      # Rohdaten (Original-CSV)
├── sql/
│   └── schema.sql                     # Datenbankschema (MySQL)
├── src/
│   ├── config.py                      # DB-Zugangsdaten & Pfade (HIER ANPASSEN)
│   ├── db.py                          # Verbindungs-Hilfsfunktionen
│   ├── notebook_helpers.py            # Hilfsfunktion zum Laden der Skripte im Notebook
│   ├── 01_load_data.py                # Rohdaten -> MySQL
│   ├── 02_clean_data.py               # Datenbereinigung
│   ├── 03_descriptive_stats.py        # Deskriptive Statistik
│   ├── 04_einflussanalyse.py          # Einflussanalyse
│   ├── 05_zusammenhangsanalyse.py     # Zusammenhangsanalyse (Korrelationen)
│   ├── 06_hypothesentests.py          # Hypothesentests & Inferenzstatistik
│   ├── 07_regression_modelle.py       # ML: Regressionsmodelle (Preisvorhersage)
│   ├── 08_klassifikation.py           # ML: Klassifikation (Bonus)
│   ├── 09_email_simulation.py         # Test-E-Mail-Generierung (Automatisierung, Bonus)
│   └── run_all.ipynb                  # Jupyter-Notebook: führt alle Schritte aus
├── output/                            # Wird automatisch befüllt:
│                                       # Diagramme (PNG), Tabellen (CSV),
│                                       # bereinigte Daten, trainierte Modelle,
│                                       # Beispiel-E-Mail
└── requirements.txt
```

Jedes Skript entspricht **genau einem Abschnitt der Aufgabendefinition** aus
der Projektpräsentation. Das erleichtert es, in der Abschlusspräsentation und
der Projektdokumentation direkt auf die jeweiligen Anforderungen Bezug zu
nehmen.

---

## Setup

### 1. Python-Umgebung

```bash
cd project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. MySQL-Datenbank einrichten

Stellen Sie sicher, dass ein lokaler MySQL-Server läuft. Führen Sie dann das
Schema aus (z.B. über die MySQL-Kommandozeile, MySQL Workbench oder phpMyAdmin):

```bash
mysql -u root -p < sql/schema.sql
```

### 3. Zugangsdaten konfigurieren

In `src/config.py` stehen die Verbindungsdaten. Empfohlen: Passwort als
Umgebungsvariable setzen, statt es im Code zu speichern:

```bash
export DB_PASSWORD="ihr_mysql_passwort"     # Linux/Mac
setx DB_PASSWORD "ihr_mysql_passwort"       # Windows (neues Terminal öffnen danach)
```

Alternativ kann das Passwort auch direkt in `config.py` eingetragen werden
(nicht empfohlen für ein GitHub-Repository – siehe Hinweis unten).

### 4. Pipeline ausführen

Empfohlen: Jupyter-Notebook, das alle Schritte nacheinander ausführt und
Diagramme direkt inline anzeigt:

```bash
cd src
jupyter notebook run_all.ipynb
```

Dann im Notebook: "Run All Cells" (bzw. Kernel -> Restart & Run All).

Alternativ können die Skripte auch einzeln über die Kommandozeile ausgeführt
werden, z.B. nur die Regressionsmodelle:

```bash
cd src
python 07_regression_modelle.py
```

**Hinweis:** Schritt 1 (`01_load_data.py`) benötigt eine funktionierende
MySQL-Verbindung. Alle folgenden Schritte (2–9) lesen die Daten aus MySQL
bzw. aus der zuvor erzeugten `output/cleaned_data.csv`.

### 5. Webinterface starten (optional)

Eine interaktive Streamlit-Anwendung zur Preis- und Kategorie-Vorhersage:

```bash
cd src
streamlit run 10_web_app.py
```

Die App öffnet sich automatisch im Browser unter **http://localhost:8501**.

**Features der Webinterface:**
- Eingabeformular mit allen 6 Features (Location, Wohnfläche, BHK, Bad, Balkon, Flächenart)
- Live-Preis-Vorhersage (Regression)
- Live-Preiskategorie-Vorhersage (Klassifikation: cheap/medium/expensive)
- Modell-Vertrauen (Wahrscheinlichkeiten für jede Kategorie)
- Kontext-Informationen (Durchschnittspreis & Preisspanne der Lage)
- Responsive, benutzerfreundliches Design

---

## Was macht jedes Skript? (Mapping zur Aufgabendefinition)

| Skript | Aufgabenbereich | Inhalt |
|---|---|---|
| `01_load_data.py` | Speicherung des Datensatzes | CSV unverändert in MySQL laden |
| `02_clean_data.py` | Datenqualität | Bereinigung `total_sqft`, fehlende Werte, Ausreißer, Feature Engineering (`bhk`, `price_per_sqft`) |
| `03_descriptive_stats.py` | Deskriptive Statistik | Lageparameter, Streuungsmaße, Verteilungen, Visualisierungen |
| `04_einflussanalyse.py` | Einflussanalyse | Einzel- und Gesamteinfluss der Variablen auf `price` (Korrelation, ANOVA, Random-Forest-Importance) |
| `05_zusammenhangsanalyse.py` | Zusammenhangsanalyse | Pearson-/Spearman-Korrelation, Cramér's V, Stärke & Richtung |
| `06_hypothesentests.py` | Hypothesentests | t-Test, ANOVA, Signifikanztest der Korrelation, inkl. Prüfung der Testvoraussetzungen |
| `07_regression_modelle.py` | Machine Learning (verpflichtend) | Vergleich von 4 Regressionsmodellen mit Robust Scaling, Cross-Validation, Testset-Evaluation, Modell wird gespeichert |
| `08_klassifikation.py` | Machine Learning (Bonus) | Preiskategorisierung (cheap/medium/expensive), Vergleich Logistische Regression vs. Decision Tree |
| `09_email_simulation.py` | Automatisierung (Bonus) | Erzeugt eine Test-E-Mail mit fiktiven Kundendaten als Beispiel für eine eingehende Anfrage |
| `10_web_app.py` | Webinterface (Bonus) | Interaktive Streamlit-Anwendung zur Preis- und Kategorie-Vorhersage mit Live-Eingabeformular |

Die optionalen Bonus-Teile (Klassifikation, E-Mail-Simulation, Webinterface) sind nun
ebenfalls enthalten. Eine echte Verarbeitung realer E-Mails (Parsing,
automatisierte Preisvorhersage direkt aus dem Text) ist
**nicht** Teil dieses Lieferumfangs und würde den nächsten Ausbauschritt
darstellen.

---

## Wichtige methodische Entscheidungen (für die Dokumentation)

- **Skalierung der numerischen Features:** Es wird `RobustScaler` (statt
  `StandardScaler`) verwendet, da `total_sqft_clean`, `bath` und `bhk` auch
  nach der Bereinigung noch eine deutliche Rechtsschiefe und einzelne
  Extremwerte aufweisen. `RobustScaler` skaliert anhand von Median und
  Interquartilsabstand (IQR) statt Mittelwert/Standardabweichung und ist
  dadurch robuster gegenüber den verbleibenden Ausreißern. `Min-Max-Scaling`
  wurde bewusst nicht verwendet, da ein einzelner Extremwert dabei die gesamte
  Skala verzerren würde.
- **`total_sqft`-Bereinigung:** Werte mit Wertebereichen (z.B. "2100 - 2850")
  wurden durch den Mittelwert ersetzt; alternative Flächeneinheiten (Sq. Meter,
  Acres, Cents, Guntha, Perch, Sq. Yards) wurden in Quadratfuß umgerechnet.
- **Ausreißerbehandlung:** Unrealistische Flächen-/Zimmerverhältnisse, ein
  oberes Plausibilitätslimit für `total_sqft_clean` (max. 20.000 sqft) sowie
  extreme `price_per_sqft`-Werte (IQR-Methode, je Location) wurden entfernt.
- **Fehlende Werte:** `bath`/`balcony` wurden gruppenweise nach BHK-Anzahl
  durch den Median ersetzt; `society` wurde als "Unknown" markiert statt
  Zeilen zu verwerfen, da es für die Kernanalyse nicht zwingend benötigt wird.
- **Seltene Locations:** Locations mit weniger als 10 Einträgen wurden zu
  "other" zusammengefasst, um Overfitting beim One-Hot-Encoding zu vermeiden.
- **Modellauswahl (Regression):** Vier Modelle wurden verglichen (Lineare
  Regression, Ridge, Random Forest, Gradient Boosting), jeweils mit 5-facher
  Cross-Validation und finaler Evaluation auf einem unabhängigen Testset
  (80/20-Split).
- **Klassengrenzen (Klassifikation):** `price` wurde anhand der Terzile
  (33%-/67%-Quantil) in drei etwa gleich große Klassen eingeteilt
  (cheap/medium/expensive), damit die Klassifikationsmetriken nicht durch
  unausgeglichene Klassen verzerrt werden. Der Split wurde zusätzlich
  stratifiziert, um diese Balance auch im Train/Test-Split zu erhalten.

---

## Ergebnisse (Kurzüberblick)

**Regression:**
- Bestes Modell: **Gradient Boosting**, Test R² ≈ **0.61**, MAE ≈ **29 Lakhs**
- Wichtigster Einflussfaktor: **Wohnfläche (total_sqft)**, gefolgt von Lage
  und Anzahl der Bäder.

**Klassifikation (Bonus):**
- Klassen: cheap (≤ 55 Lakhs) / medium (55–90 Lakhs) / expensive (> 90 Lakhs)
- Bestes Modell: **Logistische Regression**, Test Accuracy ≈ **76%**
  (Decision Tree: ≈ 71%), Zufallsniveau läge bei ca. 33%.

**Statistische Tests:**
- Alle drei durchgeführten Hypothesentests sind statistisch signifikant
  (p < 0.05).

**Automatisierung (Bonus):**
- Eine Beispiel-Kundenanfrage (`beispiel_kundenanfrage.txt`) wurde als
  Testdatei erzeugt, um zu zeigen, wie eine eingehende Anfrage aussehen
  könnte (ausschließlich fiktive Daten, siehe Datenschutz-Hinweis unten).

Alle Zahlenwerte sowie Diagramme liegen vollständig in `/output` vor und
sollten für die Präsentation/Dokumentation direkt übernommen werden.

---

## Datenschutz-Hinweis

Dieser Datensatz enthält **keine personenbezogenen Daten** über reale Käufer
oder Verkäufer, sondern ausschließlich Immobilienmerkmale (Lage, Fläche,
Preis etc.). Er ist somit unkritisch im Sinne der im Projekt geforderten
Vermeidung realer Kundendaten.

Die in `09_email_simulation.py` erzeugte Beispiel-E-Mail verwendet
ausschließlich frei erfundene Testdaten (fiktive Namen, eine ungültige
Beispiel-Domain `*.invalid`). Es handelt sich nicht um echte Kundendaten.

---

## Nächste Schritte für die Projektdokumentation

1. Die Diagramme aus `/output` direkt in die Präsentation/Dokumentation einfügen.
2. Die Tabellen (`*.csv` in `/output`) für Tabellen in der schriftlichen
   Dokumentation verwenden.
3. `sql/schema.sql` als Nachweis für das "vollständige MySQL-Datenbankschema
   inklusive SQL-Export" einreichen (ggf. um einen `mysqldump`-Export des
   befüllten Tables ergänzen).
4. Diesen gesamten `src/`-Ordner als Python-Source-Code einreichen.
