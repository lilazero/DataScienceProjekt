"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data cleaning and preprocessing for real estate data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def load_and_clean(self) -> pd.DataFrame:
        """Load and perform initial cleaning"""
        logger.info(f"Original dataset shape: {self.df.shape}")
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        logger.info(f"After removing duplicates: {self.df.shape}")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Convert and clean columns
        self._clean_columns()
        
        logger.info(f"Final cleaned shape: {self.df.shape}")
        return self.df
    
    def _handle_missing_values(self):
        """Handle missing values appropriately"""
        # For numeric columns, use median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # For categorical columns, use mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        logger.info(f"Missing values handled. Remaining NaN: {self.df.isna().sum().sum()}")
    
    def _clean_columns(self):
        """Clean individual columns"""
        # Convert 'total_sqft' to numeric
        if 'total_sqft' in self.df.columns:
            self.df['total_sqft'] = pd.to_numeric(
                self.df['total_sqft'], 
                errors='coerce'
            )
        
        # Convert 'price' to numeric if needed
        if 'price' in self.df.columns:
            self.df['price'] = pd.to_numeric(
                self.df['price'], 
                errors='coerce'
            )
        
        # Clean numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
    
    def get_statistics(self) -> dict:
        """Generate statistical summary"""
        stats = {
            'shape': self.df.shape,
            'missing_values': self.df.isna().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_summary': self.df.describe().to_dict()
        }
        return stats


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file"""
    return pd.read_csv(filepath)


def split_features_target(df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
