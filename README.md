# Real Estate Price Prediction Project - DataScienceProjekt

## Project Overview

This is a comprehensive Data Science project focused on predicting real estate property prices in Bengaluru, India. The project combines data analysis, machine learning, and deep learning techniques to build predictive models for house prices based on various property features.

## Project Objectives

- **Data Import & Exploration**: Load and analyze the Bengaluru house price dataset
- **Exploratory Data Analysis (EDA)**: Perform statistical analysis and identify patterns in property data
- **Machine Learning**: Develop and train multiple ML models to predict house prices
- **Deep Learning**: Implement neural networks for enhanced prediction capabilities
- **Database Integration**: Store and manage property listings using MySQL database

## Project Structure

```
DataScienceProjekt/
├── data/
│   └── raw/
│       └── Bengaluru_House_Data.csv       # Source dataset
├── notebooks/
│   ├── 1_data_import.ipynb                # Data loading and cleaning
│   ├── 2_eda_and_statistics.ipynb         # Exploratory data analysis
│   ├── 3_machine_learning.ipynb           # ML model development
│   └── 4_deep_learning_nip.ipynb          # Deep learning models
├── sql/
│   ├── database_views.sql                 # Database views and queries
│   └── real_estate_price_prediction_db.sql# Database schema
├── src/
│   ├── data_preprocessing.py              # Data cleaning utilities
│   ├── db_connector.py                    # MySQL database connector
│   ├── email_parser.py                    # Email parsing utilities
│   ├── train_model.py                     # Model training scripts
│   ├── db_credentials.py                  # Database credentials (PRIVATE - DO NOT COMMIT)
│   └── db_credentials_example.py          # Example credentials template
└── models/                                 # Trained model storage

```

## Technologies & Libraries

- **Python 3.x**
- **Data Science**: pandas, NumPy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Deep Learning**: TensorFlow/Keras
- **Database**: MySQL, mysql-connector-python
- **Utilities**: python-dotenv

## Installation & Setup

### 1. Prerequisites

- Python 3.7 or higher
- MySQL Server installed and running
- Virtual environment (recommended)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Database Configuration

The project uses MySQL to store and manage property listings. Follow these steps to set up the database connection:

#### Step 1: Create Database Credentials File

You must create a `src/db_credentials.py` file with your actual database credentials. This file is **private** and should **never** be committed to version control.

**DO NOT use db_credentials.py from version control. Always create your own!**

1. Copy the example template:

   ```bash
   cp src/db_credentials_example.py src/db_credentials.py
   ```

2. Edit `src/db_credentials.py` with your actual MySQL credentials:
   ```python
   # Database Credentials Configuration
   DB_HOST = "your_mysql_host"      # e.g., "localhost" or "127.0.0.1"
   DB_USER = "your_mysql_user"      # e.g., "root"
   DB_PASSWORD = "your_mysql_password"  # Your MySQL password
   DB_NAME = "your_database_name"   # e.g., "yay_studios_real_estate"
   DB_PORT = 3306                   # Default MySQL port (or 3307 for custom)
   ```

#### Step 2: Ensure .gitignore is Configured

The `.gitignore` file already includes `db_credentials.py` to prevent accidental commits of sensitive data.

#### Step 3: Test Database Connection

```bash
cd src
python db_connector.py
```

Expected output if connection is successful:

```
INFO - Successfully connected to MySQL Server version X.X.X
INFO - Connected to database: your_database_name
INFO - Table stats for property_listings: X rows, 11 columns
```

### 4. Create MySQL Database (Optional)

If the database doesn't exist, the connector can create it:

```python
from src.db_connector import MySQLConnector

connector = MySQLConnector(
    host="localhost",
    user="root",
    password="your_password",
    database="yay_studios_real_estate",
    port=3306
)

connector.create_database_if_not_exists()
```

## Usage

### Running Notebooks

1. **Data Import**: `notebooks/1_data_import.ipynb`
   - Loads CSV data and performs initial data cleaning

2. **EDA & Statistics**: `notebooks/2_eda_and_statistics.ipynb`
   - Analyzes data distributions and relationships

3. **Machine Learning**: `notebooks/3_machine_learning.ipynb`
   - Trains traditional ML models (Linear Regression, Random Forest, etc.)

4. **Deep Learning**: `notebooks/4_deep_learning_nip.ipynb`
   - Implements neural network models

### Using the Database Connector

```python
from src.db_connector import MySQLConnector

# Create connector with credentials
connector = MySQLConnector()

# Connect to database
if connector.connect():
    # Load data from CSV
    connector.load_csv_to_db('data/raw/Bengaluru_House_Data.csv')

    # Fetch data as DataFrame
    df = connector.get_data_as_dataframe()
    print(df.head())

    # Disconnect
    connector.disconnect()
```

## Features

### Database Connector Features

- Connection pooling support
- CSV file loading to database
- Data fetching as pandas DataFrame
- Custom SQL query execution
- Table statistics and information
- Context manager support (`with` statement)
- Comprehensive logging

### Example Usage Patterns

```python
# Context manager usage (auto-disconnect)
with MySQLConnector() as connector:
    df = connector.get_data_as_dataframe()

# Custom SQL queries
results = connector.fetch_query_results(
    "SELECT * FROM property_listings WHERE price > %s",
    (1000000,)
)

# Get table statistics
stats = connector.get_table_stats()
print(f"Total records: {stats['row_count']}")
print(f"Columns: {stats['columns']}")
```

## Important Notes

⚠️ **Security Notice**:

- **Never commit `db_credentials.py` to version control**
- Always use `db_credentials_example.py` as a template
- Add `db_credentials.py` to `.gitignore` (already configured)
- Consider using environment variables for production environments

## Troubleshooting

### Connection Issues

**Error**: `Access denied for user 'root'@'localhost'`

- Check your database credentials in `src/db_credentials.py`
- Verify MySQL server is running
- Ensure the user has correct permissions

**Error**: `mysql.connector.errors.DatabaseError: Unknown database`

- The database might not exist
- Run `connector.create_database_if_not_exists()`

**Error**: `ModuleNotFoundError: No module named 'mysql'`

- Install mysql-connector-python: `pip install mysql-connector-python`

## Future Enhancements

- [ ] API endpoint for predictions
- [ ] Dashboard for visualization
- [ ] Model deployment and serving
- [ ] Extended feature engineering
- [ ] Cross-validation and hyperparameter tuning
- [ ] Production-ready error handling

## License

See [LICENSE](LICENSE) file for details.

## Contributors

- Data Science Project Team

---

**Last Updated**: 2026-06-15
