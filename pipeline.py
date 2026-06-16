"""
Main Pipeline Script
Orchestrates the full data science workflow
"""

import sys
import pandas as pd
sys.path.insert(0, 'src')

from data_preprocessing import DataPreprocessor, load_csv, split_features_target
from train_model import HousePricePredictor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution"""
    
    logger.info("=" * 70)
    logger.info("BENGALURU HOUSE PRICE PREDICTION - FULL PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Load data
    logger.info("\n[1/5] Loading data...")
    df = load_csv('data/raw/Bengaluru_House_Data.csv')
    logger.info(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Preprocess data
    logger.info("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor(df)
    df_clean = preprocessor.load_and_clean()
    logger.info(f"✓ Cleaned to {df_clean.shape[0]} rows")
    
    # Step 3: Split features and target
    logger.info("\n[3/5] Preparing features...")
    X, y = split_features_target(df_clean, target_col='price')
    logger.info(f"✓ Features: {X.shape[1]}, Target: 1")
    
    # Step 4: Train models
    logger.info("\n[4/5] Training models...")
    predictor = HousePricePredictor()
    predictor.preprocess_data(X, y, test_size=0.2)
    predictor.train_models()
    logger.info("✓ 3 models trained (Linear Regression, Random Forest, Gradient Boosting)")
    
    # Step 5: Evaluate models
    logger.info("\n[5/5] Evaluating models...")
    results = predictor.evaluate_models()
    
    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  R² Score: {metrics['R2']:.4f}")
        logger.info(f"  RMSE: {metrics['RMSE']:.2f}")
        logger.info(f"  MAE: {metrics['MAE']:.2f}")
    
    best_model_name, best_model = predictor.get_best_model()
    logger.info(f"\n✓ Best Model: {best_model_name}")
    logger.info("\n✓ Pipeline completed successfully!")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
