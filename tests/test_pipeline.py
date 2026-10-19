"""
Unit Tests for Maritime Port Congestion Prediction Pipeline
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data_generator import generate_synthetic_port_dataset
from src.preprocessing import MaritimeFeatureEngineer, create_preprocessor
from src.model import PortCongestionPredictor
from src.explainability import MaritimeExplainer
from src.recommendations import ActionableRecommender


def test_data_generation():
    df = generate_synthetic_port_dataset(n_samples=50, random_seed=123)
    assert len(df) == 50
    assert "waiting_time_hours" in df.columns
    assert "is_congested_delayed" in df.columns
    assert df["waiting_time_hours"].min() >= 0.0
    assert set(df["is_congested_delayed"].unique()).issubset({0, 1})
    assert not df.isnull().any().any()


def test_feature_engineering():
    df = generate_synthetic_port_dataset(n_samples=20, random_seed=123)
    engineer = MaritimeFeatureEngineer()
    df_transformed = engineer.transform(df)
    
    assert "traffic_to_berth_ratio" in df_transformed.columns
    assert "weather_severity_index" in df_transformed.columns
    assert "estimated_hours_to_arrival" in df_transformed.columns
    assert "berth_pressure_index" in df_transformed.columns
    assert "sin_arrival_hour" in df_transformed.columns
    assert "cos_arrival_hour" in df_transformed.columns


def test_model_inference():
    predictor_path = os.path.join("models", "predictor.joblib")
    preproc_path = os.path.join("models", "preprocessor.joblib")
    assert os.path.exists(predictor_path), "Predictor model file missing"
    assert os.path.exists(preproc_path), "Preprocessor file missing"
    
    predictor = PortCongestionPredictor.load(predictor_path)
    bundle = joblib.load(preproc_path)
    
    df = generate_synthetic_port_dataset(n_samples=5, random_seed=999)
    df_eng = bundle["feature_engineer"].transform(df)
    X = bundle["col_transformer"].transform(df_eng)
    
    probs, preds, confs, tiers = predictor.predict_likelihood_and_confidence(X)
    delay_hours = predictor.predict_delay_hours(X)
    
    assert len(probs) == 5
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert (confs >= 50.0).all() and (confs <= 100.0).all()
    assert len(tiers) == 5
    assert (delay_hours >= 0.0).all()


def test_shap_explanation():
    predictor_path = os.path.join("models", "predictor.joblib")
    preproc_path = os.path.join("models", "preprocessor.joblib")
    
    predictor = PortCongestionPredictor.load(predictor_path)
    bundle = joblib.load(preproc_path)
    
    df = generate_synthetic_port_dataset(n_samples=1, random_seed=42)
    df_eng = bundle["feature_engineer"].transform(df)
    X = bundle["col_transformer"].transform(df_eng)
    
    explainer = MaritimeExplainer(predictor, bundle["feature_names"])
    explanation = explainer.explain_instance(X[0])
    
    assert "top_risk_drivers" in explanation
    assert "top_mitigating_factors" in explanation
    assert "narrative" in explanation
    assert len(explanation["top_risk_drivers"]) > 0


def test_actionable_recommendations():
    recommender = ActionableRecommender()
    
    # Severe congestion case
    actions = recommender.generate_recommendations(
        prediction_prob=0.88,
        predicted_delay_hours=18.5,
        top_risk_drivers=[{"feature": "anchorage_queue_length", "shap_value": 0.35}],
        raw_inputs={
            "ais_speed_knots": 16.0,
            "ais_distance_nm": 65.0,
            "anchorage_queue_length": 14,
            "berth_occupancy_ratio": 0.95,
            "wind_speed_knots": 34.0,
            "wave_height_m": 3.1,
            "labor_status": "Reduced Shift",
            "customs_inspection_required": 1
        }
    )
    
    assert len(actions) >= 3
    categories = [a["category"] for a in actions]
    assert any("Virtual Arrival" in c for c in categories)
    assert any("Adverse Weather" in c for c in categories)
    assert any("Regulatory" in c for c in categories)
