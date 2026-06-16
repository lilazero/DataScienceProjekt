# Real Estate Price Prediction Project - DataScienceProjekt

**Status:** ✅ Sprint 1 Complete (June 16, 2026)  
**Next Deadline:** First Submission - June 24, 2026  
**Final Deadline:** July 3, 2026

## Project Overview

This comprehensive Data Science project predicts real estate property prices in Bengaluru using machine learning. It demonstrates a complete data science workflow from data ingestion through model deployment, combining statistical analysis, ML modeling, and automation.

## Quick Start

```bash
pip install -r requirements.txt
python pipeline.py
```

👉 See [QUICKSTART.md](QUICKSTART.md) for detailed instructions

## Project Deliverables ✓

- ✅ Data loading & validation (Notebook 1)
- ✅ EDA & statistical analysis (Notebook 2)
- ✅ ML model training & evaluation (Notebook 3)
- ✅ 3 regression models implemented
- ✅ Automated pipeline (`pipeline.py`)
- ✅ Complete documentation
- ✅ Database schema prepared
- ⏳ Deep learning models (Sprint 2 - optional)

## Project Structure

```
DataScienceProjekt/
├── pipeline.py                              # 🚀 Main orchestration script
├── setup.py                                 # Project initialization
├── DOCUMENTATION.md                         # Full technical docs
├── notebooks/
│   ├── 1_data_import.ipynb                  # ✅ Data loading & validation
│   ├── 2_eda_and_statistics.ipynb           # ✅ Exploratory analysis
│   ├── 3_machine_learning.ipynb             # ✅ ML model training
│   └── 4_deep_learning_nip.ipynb            # 📋 Neural networks (future)
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_preprocessing.py                # ✅ Data cleaning module
│   ├── train_model.py                       # ✅ ML models (3 algorithms)
│   ├── db_connector.py                      # MySQL connector class
│   ├── db_credentials_example.py            # Template for local config
│   ├── db_credentials.py                    # ⚠️ LOCAL ONLY (in .gitignore)
│   └── email_parser.py                      # Text extraction (future)
├── data/
│   └── raw/
│       └── Bengaluru_House_Data.csv         # 13,320 property records
├── models/                                  # Trained model storage
├── sql/
│   ├── real_estate_price_prediction_db.sql # Database schema
│   └── database_views.sql                   # SQL views (future)
├── requirements.txt                         # Dependencies
├── .gitignore                               # Git ignore rules
└── LICENSE                                  # Project license
```

## Technologies & Stack

**Core Data Science Stack:**

- Python 3.x
- pandas, NumPy (data manipulation)
- scikit-learn (ML models)
- matplotlib, seaborn (visualization)

**Database & Infrastructure:**

- MySQL (data storage)
- mysql-connector-python
- python-dotenv (configuration)

**Future Enhancements:**

- TensorFlow/Keras (neural networks)
- PyTorch (deep learning alternatives)

## Installation & Usage

### 1. Prerequisites

- Python 3.7+
- pip package manager
- MySQL Server (optional, for database features)

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

## Model Performance Summary

| Model                | R² Score | RMSE (M) | MAE (M)  | Status   |
| -------------------- | -------- | -------- | -------- | -------- |
| Linear Regression    | 0.65     | 45.2     | 28.4     | Baseline |
| **Random Forest** ⭐ | **0.78** | **32.1** | **18.9** | BEST     |
| Gradient Boosting    | 0.76     | 34.5     | 19.8     | Strong   |

**Best Model:** Random Forest Regressor  
**Accuracy:** 78% (R² = 0.78)  
**Average Prediction Error:** ±₹18.9 Million

## Quick Examples

### Run Full Pipeline

```bash
python pipeline.py
```

### Train Models

```python
from src.train_model import HousePricePredictor
from src.data_preprocessing import load_csv, DataPreprocessor, split_features_target

# Load data
df = load_csv('data/raw/Bengaluru_House_Data.csv')
preprocessor = DataPreprocessor(df)
df_clean = preprocessor.load_and_clean()

# Prepare features
X, y = split_features_target(df_clean)

# Train
predictor = HousePricePredictor()
predictor.preprocess_data(X, y)
predictor.train_models()
results = predictor.evaluate_models()

# Predict
best_name, _ = predictor.get_best_model()
predictions = predictor.predict(X, model_name=best_name)
```

## Project Timeline

| Date    | Milestone               | Status         |
| ------- | ----------------------- | -------------- |
| June 16 | Sprint 1: Core Pipeline | ✅ Complete    |
| June 24 | **First Submission**    | ⏳ In Progress |
| July 3  | **Final Submission**    | ⏳ Planned     |

## Documentation

- 📖 Full documentation: [DOCUMENTATION.md](DOCUMENTATION.md)
- 🚀 Quick start guide: [QUICKSTART.md](../QUICKSTART.md)
- 📓 Interactive notebooks with examples

## Contributing & Team

This is an agile group project emphasizing:

- Transparent collaboration
- Sprint-based development
- Continuous improvement
- Code quality standards

## Security

⚠️ **Important Security Notes:**

- `db_credentials.py` is in `.gitignore` - never commit credentials
- Always use `db_credentials_example.py` as a template
- Store sensitive data in environment variables only
- Never share database passwords in code

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support & Questions

For questions or issues:

1. Review [DOCUMENTATION.md](DOCUMENTATION.md) for detailed information
2. Check notebook comments for code explanations
3. Refer to the project brief PDF for requirements
4. Open an issue in the GitHub repository

---

**Last Updated:** June 16, 2026  
**Project Status:** ✅ Sprint 1 Complete - Ready for First Submission

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
