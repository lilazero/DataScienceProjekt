"""
Database Credentials Configuration - EXAMPLE TEMPLATE
=====================================================

IMPORTANT: This is an example/template file. DO NOT use this file directly for your credentials.

SETUP INSTRUCTIONS:
1. Copy this file: cp db_credentials_example.py db_credentials.py
2. Edit the new db_credentials.py file with YOUR ACTUAL credentials
3. The db_credentials.py file is already in .gitignore and will never be committed
4. Your actual credentials file will be created locally only

SECURITY WARNING:
- Never commit db_credentials.py to version control
- Never share your credentials file
- Keep your MySQL password secure
- Use strong passwords for your MySQL user
"""

# ========== DATABASE CREDENTIALS ==========
# Edit these values with your actual MySQL credentials

# MySQL Server Host/IP Address
# Examples: "localhost", "127.0.0.1", "192.168.1.100", "db.example.com"
DB_HOST = "localhost"

# MySQL Username
# Examples: "root", "developer", "app_user"
DB_USER = "root"

# MySQL Password
# WARNING: This should be strong and kept secure
# Keep this value confidential
DB_PASSWORD = ""

# Database Name
# Examples: "real_estate_db", "yay_studios_real_estate", "bengaluru_houses"
DB_NAME = "real_estate_db"

# MySQL Port
# Default MySQL port: 3306
# Custom ports: 3307, 3308, etc.
DB_PORT = 3306

# ========== END OF CONFIGURATION ==========
