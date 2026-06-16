#!/usr/bin/env python
"""
Project Setup Script
Validates environment and initializes project directories
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'src/data_preprocessing.py',
        'src/db_connector.py',
        'src/train_model.py',
        'data/raw/Bengaluru_House_Data.csv',
        'notebooks/1_data_import.ipynb',
        'notebooks/2_eda_and_statistics.ipynb',
        'notebooks/3_machine_learning.ipynb',
        'sql/real_estate_price_prediction_db.sql',
        'requirements.txt'
    ]
    
    print("Checking project structure...")
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if key Python packages are installed"""
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    
    print("\nChecking Python dependencies...")
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (NOT INSTALLED)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def create_directories():
    """Ensure all required directories exist"""
    directories = ['data/raw', 'models', 'notebooks', 'sql', 'src']
    
    print("\nCreating directories...")
    for dir in directories:
        Path(dir).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir}")

def main():
    print("=" * 60)
    print("Bengaluru House Price Prediction - Project Setup")
    print("=" * 60)
    
    create_directories()
    
    files_ok = check_requirements()
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    if files_ok and deps_ok:
        print("✓ Project setup complete! Ready to run.")
        print("\nNext steps:")
        print("1. Configure db_credentials.py with MySQL settings")
        print("2. Run: python pipeline.py")
        print("3. Or open notebooks in Jupyter")
    else:
        print("⚠ Some setup steps are incomplete.")
        print("Please review the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
