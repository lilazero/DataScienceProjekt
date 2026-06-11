-- 1. Create the database for the project (if it does not exist yet)
CREATE DATABASE IF NOT EXISTS yay_studios_real_estate;
USE yay_studios_real_estate;

-- 2. Drop the table if it already exists (for a clean reset)
-- DROP TABLE IF EXISTS property_listings;

-- 3. Create the table matching our CSV file structure
CREATE TABLE property_listings (
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

-- 4. Verify that the table was successfully created and is empty
SELECT * FROM property_listings;