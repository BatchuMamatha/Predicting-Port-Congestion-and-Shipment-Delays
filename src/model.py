"""
Model Architecture: Dual-Head Predictor (Congestion Classification + Delay Regression)
Includes Probability Calibration, Confidence Scoring, and Baseline Benchmarking.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
import joblib


class PortCongestionPredictor:
    """
    Dual-head prediction system for maritime logistics:
    1. Calibrated Classifier: Predicts P(Congestion Delay >= Threshold).
    2. Continuous Regressor: Predicts expected waiting time in hours.
    3. Confidence Scorer: Quantifies certainty based on probability margins and calibration.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # Base estimators
        self.base_classifier = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state
        )
        
        # Calibrated classifier for reliable probabilistic outputs
        self.calibrated_classifier: Optional[CalibratedClassifierCV] = None
        
        # Continuous delay hours regressor
        self.regressor = XGBRegressor(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state
        )
        
        # Linear Baseline models for comparative validation
        self.baseline_classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        self.baseline_regressor = Ridge(alpha=1.0)
        
        self.feature_names: list = []

    def fit(self, X_train: np.ndarray, y_class_train: np.ndarray, y_reg_train: np.ndarray, feature_names: list = None):
        """
        Train both classification and regression heads along with baselines.
        """
        self.feature_names = feature_names if feature_names is not None else []
        
        # 1. Fit Base Classifier & Calibrate
        self.base_classifier.fit(X_train, y_class_train)
        self.calibrated_classifier = CalibratedClassifierCV(
            estimator=self.base_classifier,
            method="sigmoid",
            cv="prefit"
        )
        self.calibrated_classifier.fit(X_train, y_class_train)
        
        # 2. Fit Continuous Regressor
        self.regressor.fit(X_train, y_reg_train)
        
        # 3. Fit Baselines
        self.baseline_classifier.fit(X_train, y_class_train)
        self.baseline_regressor.fit(X_train, y_reg_train)

    def predict_likelihood_and_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Returns:
        - probs: Probabilities of delay [0, 1]
        - binary_preds: 0 or 1
        - confidence_scores: Normalized confidence score percentage [50.0, 99.0]
        - confidence_tiers: 'High', 'Moderate', 'Low'
        """
        if self.calibrated_classifier is None:
            raise ValueError("Model is not fitted yet.")
            
        probs = self.calibrated_classifier.predict_proba(X)[:, 1]
        binary_preds = (probs >= 0.50).astype(int)
        
        # Margin from decision boundary (|P - 0.5| * 2) in [0, 1]
        margin = np.abs(probs - 0.50) * 2.0
        
        # Smooth confidence curve: maps 0 margin to 52%, 1.0 margin to 99%
        confidence_scores = np.round(50.0 + (margin ** 0.8) * 49.0, 1)
        
        confidence_tiers = []
        for score in confidence_scores:
            if score >= 82.0:
                confidence_tiers.append("High Confidence")
            elif score >= 65.0:
                confidence_tiers.append("Moderate Confidence")
            else:
                confidence_tiers.append("Low / Equivocal")
                
        return probs, binary_preds, confidence_scores, confidence_tiers

    def predict_delay_hours(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts continuous waiting time in hours (clipped to >= 0.0).
        """
        preds = self.regressor.predict(X)
        return np.maximum(preds, 0.0)

    def save(self, filepath: str):
        """Save model bundle to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PortCongestionPredictor":
        """Load model bundle from disk."""
        return joblib.load(filepath)
