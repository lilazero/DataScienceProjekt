-- ============================================================
-- Datenbankschema: yay_studios_real_estate
-- Tabelle: property_listings
-- Quelle: Bengaluru House Price Dataset (Kaggle)
-- ============================================================

-- 1. Datenbank erstellen (falls noch nicht vorhanden)
CREATE DATABASE IF NOT EXISTS yay_studios_real_estate;
USE yay_studios_real_estate;

-- 2. Tabelle bei Bedarf zurücksetzen (für einen sauberen Neustart)
-- DROP TABLE IF EXISTS property_listings;

-- 3. Tabelle passend zur CSV-Struktur anlegen
CREATE TABLE IF NOT EXISTS property_listings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    area_type VARCHAR(50) NOT NULL,
    availability VARCHAR(50) NOT NULL,
    location VARCHAR(255),
    size VARCHAR(50),
    society VARCHAR(100),
    total_sqft VARCHAR(50) NOT NULL,
    bath INT,
    balcony INT,
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Prüfen, ob die Tabelle korrekt erstellt wurde
SELECT * FROM property_listings LIMIT 10;

-- Hinweis: total_sqft bleibt bewusst VARCHAR, da die Rohdaten
-- Wertebereiche (z.B. "2100 - 2850") und Einheiten (z.B. "34.46Sq. Meter")
-- enthalten. Die numerische Bereinigung erfolgt in Python (siehe
-- src/01_load_data.py), bevor die Daten für die Analyse verwendet werden.
