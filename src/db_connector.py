"""
MySQL Database Connector for Real Estate Price Prediction Project
Handles all database operations including data loading, querying, and management
"""

import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
import logging
from typing import Optional, List, Dict, Tuple
import os
from dotenv import load_dotenv
from db_credentials import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MySQLConnector:
    """
    MySQL Database Connector for the Real Estate Price Prediction Project.
    Manages connections, data loading, and database operations.
    """

    def __init__(
        self,
        host: str = DB_HOST,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
        database: str = DB_NAME,
        port: int = DB_PORT
    ):
        """
        Initialize MySQL connector with connection parameters.

        Args:
            host: MySQL server host (default: localhost)
            user: MySQL username (default: root)
            password: MySQL password (default: empty)
            database: Database name (default: real_estate_db)
            port: MySQL port (default: 3307)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """
        Establish connection to MySQL database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )

            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                db_info = self.connection.get_server_info()
                logger.info(f"Successfully connected to MySQL Server version {db_info}")
                logger.info(f"Connected to database: {self.database}")
                return True
        except Error as e:
            logger.error(f"Error while connecting to MySQL: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Close database connection.

        Returns:
            bool: True if disconnected successfully
        """
        try:
            if self.connection and self.connection.is_connected():
                self.cursor.close()
                self.connection.close()
                logger.info("MySQL connection closed")
                return True
        except Error as e:
            logger.error(f"Error while disconnecting: {e}")
            return False

    def create_database_if_not_exists(self) -> bool:
        """
        Create database if it doesn't exist.

        Returns:
            bool: True if successful
        """
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port
            )
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.database}"
            )
            logger.info(f"Database '{self.database}' ensured to exist")
            cursor.close()
            conn.close()
            return True
        except Error as e:
            logger.error(f"Error creating database: {e}")
            return False

    def load_csv_to_db(self, csv_path: str, table_name: str = "property_listings") -> bool:
        """
        Load CSV file into database table.

        Args:
            csv_path: Path to CSV file
            table_name: Target table name (default: property_listings)

        Returns:
            bool: True if successful
        """
        try:
            if not self.connection or not self.connection.is_connected():
                logger.error("Database connection not established")
                return False

            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows from {csv_path}")

            # Prepare data for insertion
            columns = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

            # Convert dataframe to list of tuples
            data_tuples = [tuple(row) for row in df.values]

            # Insert data
            self.cursor.executemany(insert_query, data_tuples)
            self.connection.commit()

            logger.info(f"Successfully inserted {self.cursor.rowcount} rows into {table_name}")
            return True

        except Error as e:
            logger.error(f"Error loading CSV to database: {e}")
            self.connection.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def get_data_as_dataframe(
        self,
        query: Optional[str] = None,
        table_name: str = "property_listings"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from database and return as pandas DataFrame.

        Args:
            query: Custom SQL query. If None, selects all from table_name
            table_name: Table name to query (used if query is None)

        Returns:
            pd.DataFrame: Data from database, or None if error
        """
        try:
            if not self.connection or not self.connection.is_connected():
                logger.error("Database connection not established")
                return None

            if query is None:
                query = f"SELECT * FROM {table_name}"

            df = pd.read_sql(query, self.connection)
            logger.info(f"Successfully fetched {len(df)} rows")
            return df

        except Error as e:
            logger.error(f"Error fetching data: {e}")
            return None

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> bool:
        """
        Execute a custom SQL query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query to execute
            params: Query parameters (for parameterized queries)

        Returns:
            bool: True if successful
        """
        try:
            if not self.connection or not self.connection.is_connected():
                logger.error("Database connection not established")
                return False

            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            self.connection.commit()
            logger.info(f"Query executed successfully. Rows affected: {self.cursor.rowcount}")
            return True

        except Error as e:
            logger.error(f"Error executing query: {e}")
            self.connection.rollback()
            return False

    def fetch_query_results(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Tuple]]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL SELECT query
            params: Query parameters

        Returns:
            List[Tuple]: Query results, or None if error
        """
        try:
            if not self.connection or not self.connection.is_connected():
                logger.error("Database connection not established")
                return None

            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            results = self.cursor.fetchall()
            logger.info(f"Fetched {len(results)} rows")
            return results

        except Error as e:
            logger.error(f"Error fetching results: {e}")
            return None

    def get_table_stats(self, table_name: str = "property_listings") -> Optional[Dict]:
        """
        Get basic statistics about a table.

        Args:
            table_name: Table name

        Returns:
            Dict with table statistics
        """
        try:
            if not self.connection or not self.connection.is_connected():
                logger.error("Database connection not established")
                return None

            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            self.cursor.execute(count_query)
            row_count = self.cursor.fetchone()[0]

            # Get table structure
            structure_query = f"DESCRIBE {table_name}"
            self.cursor.execute(structure_query)
            columns = self.cursor.fetchall()

            stats = {
                "table_name": table_name,
                "row_count": row_count,
                "columns": [col[0] for col in columns],
                "total_columns": len(columns)
            }

            logger.info(f"Table stats for {table_name}: {row_count} rows, {len(columns)} columns")
            return stats

        except Error as e:
            logger.error(f"Error getting table stats: {e}")
            return None

    def clear_table(self, table_name: str = "property_listings") -> bool:
        """
        Clear all data from a table (DELETE operation).

        Args:
            table_name: Table name to clear

        Returns:
            bool: True if successful
        """
        try:
            if not self.connection or not self.connection.is_connected():
                logger.error("Database connection not established")
                return False

            query = f"DELETE FROM {table_name}"
            self.cursor.execute(query)
            self.connection.commit()
            logger.warning(f"Cleared table {table_name}. Rows deleted: {self.cursor.rowcount}")
            return True

        except Error as e:
            logger.error(f"Error clearing table: {e}")
            self.connection.rollback()
            return False

    def close(self):
        """Alias for disconnect()"""
        self.disconnect()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Example usage
if __name__ == "__main__":
    # Create connector instance
    connector = MySQLConnector()

    # Connect to database
    if connector.connect():
        # Get table statistics
        stats = connector.get_table_stats()
        if stats:
            print(f"\nTable Statistics:")
            print(f"Table: {stats['table_name']}")
            print(f"Rows: {stats['row_count']}")
            print(f"Columns: {stats['total_columns']}")

        # Example: Fetch data
        df = connector.get_data_as_dataframe()
        if df is not None:
            print(f"\nData shape: {df.shape}")
            print(df.head())

        # Disconnect
        connector.disconnect()
