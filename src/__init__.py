"""
Package initialization for DataScienceProjekt
"""

from .data_preprocessing import DataPreprocessor, load_csv, split_features_target
from .train_model import HousePricePredictor

__version__ = "1.0.0"
__all__ = [
    'DataPreprocessor',
    'load_csv',
    'split_features_target',
    'HousePricePredictor'
]
