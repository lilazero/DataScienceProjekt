"""
Machine Learning Model Training Module
Trains and evaluates regression models for house price prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import pickle
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """Main model training and prediction class"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Preprocess data and split into train/test"""
        
        # Encode categorical variables
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Data split: Train {self.X_train.shape}, Test {self.X_test.shape}")
    
    def train_models(self) -> Dict:
        """Train multiple regression models"""
        
        # 1. Linear Regression
        logger.info("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr
        
        # 2. Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # 3. Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        
        return self.models
    
    def evaluate_models(self) -> Dict:
        """Evaluate all trained models"""
        
        for model_name, model in self.models.items():
            if model_name == 'Linear Regression':
                X_test_eval = self.X_test_scaled
            else:
                X_test_eval = self.X_test
            
            # Predictions
            y_pred = model.predict(X_test_eval)
            
            # Metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"  R²: {r2:.4f}")
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, object]:
        """Get the best performing model"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['R2'])
        best_model = self.models[best_model_name]
        return best_model_name, best_model
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to file"""
        if model_name in self.models:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            logger.info(f"Model {model_name} saved to {filepath}")
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions on new data"""
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        model = self.models[model_name]
        
        # Encode categorical variables using fitted encoders
        X_encoded = X.copy()
        for col in self.label_encoders:
            if col in X_encoded.columns:
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
        
        # Scale if Linear Regression
        if model_name == 'Linear Regression':
            X_encoded = self.scaler.transform(X_encoded)
        
        return model.predict(X_encoded)
