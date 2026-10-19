"""
Feature Engineering and Preprocessing Pipeline for Maritime Congestion Prediction
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class MaritimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Maritime domain feature engineering transformer.
    Computes queue ratios, weather severity indices, estimated arrival hours,
    and cyclical temporal encodings.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Traffic to Operational Capacity Ratio
        operational_berths_safe = np.maximum(df["operational_berths"], 1)
        df["traffic_to_berth_ratio"] = (
            df["anchorage_queue_length"] + (df["inbound_traffic_density_24h"] * 0.4)
        ) / operational_berths_safe

        # 2. Weather Severity Index (Composite risk score 0 to 1.5)
        wind_component = np.clip(df["wind_speed_knots"] / 45.0, 0, 1.5) * 0.45
        wave_component = np.clip(df["wave_height_m"] / 4.0, 0, 1.5) * 0.35
        vis_component = np.clip((6.0 - df["visibility_nm"]) / 5.0, 0, 1.0) * 0.20
        df["weather_severity_index"] = wind_component + wave_component + vis_component

        # 3. Estimated Steaming Hours to Port Limit
        speed_safe = np.maximum(df["ais_speed_knots"], 2.0)
        df["estimated_hours_to_arrival"] = df["ais_distance_nm"] / speed_safe

        # 4. Berth Congestion Pressure Index
        gate_factor = 1.0 + (df["gate_congestion_index"] / 100.0)
        df["berth_pressure_index"] = df["berth_occupancy_ratio"] * gate_factor

        # 5. Crane-to-Queue Bottleneck
        crane_safe = np.maximum(df["crane_availability_ratio"], 0.2)
        df["queue_per_crane_ratio"] = df["anchorage_queue_length"] / (operational_berths_safe * crane_safe)

        # 6. Cyclical Temporal Encodings (preserve 24h & 7d continuity)
        df["sin_arrival_hour"] = np.sin(2 * np.pi * df["scheduled_arrival_hour"] / 24.0)
        df["cos_arrival_hour"] = np.cos(2 * np.pi * df["scheduled_arrival_hour"] / 24.0)
        df["sin_arrival_day"] = np.sin(2 * np.pi * df["scheduled_arrival_dayofweek"] / 7.0)
        df["cos_arrival_day"] = np.cos(2 * np.pi * df["scheduled_arrival_dayofweek"] / 7.0)
        df["sin_arrival_month"] = np.sin(2 * np.pi * df["scheduled_arrival_month"] / 12.0)
        df["cos_arrival_month"] = np.cos(2 * np.pi * df["scheduled_arrival_month"] / 12.0)

        return df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Returns lists of numerical and categorical feature names used by models.
    """
    numerical_cols = [
        "vessel_teu_capacity",
        "vessel_draft_m",
        "total_berths",
        "operational_berths",
        "crane_availability_ratio",
        "berth_occupancy_ratio",
        "anchorage_queue_length",
        "inbound_traffic_density_24h",
        "gate_congestion_index",
        "ais_distance_nm",
        "ais_speed_knots",
        "wind_speed_knots",
        "wave_height_m",
        "visibility_nm",
        "is_adverse_weather",
        "customs_inspection_required",
        "pilot_dispatch_delay_hours",
        "is_peak_season",
        "traffic_to_berth_ratio",
        "weather_severity_index",
        "estimated_hours_to_arrival",
        "berth_pressure_index",
        "queue_per_crane_ratio",
        "sin_arrival_hour",
        "cos_arrival_hour",
        "sin_arrival_day",
        "cos_arrival_day",
        "sin_arrival_month",
        "cos_arrival_month",
    ]

    categorical_cols = [
        "port_name",
        "vessel_type",
        "labor_status",
    ]

    return numerical_cols, categorical_cols


def create_preprocessor() -> Tuple[MaritimeFeatureEngineer, ColumnTransformer]:
    """
    Builds the complete preprocessing pipeline.
    """
    engineer = MaritimeFeatureEngineer()
    numerical_cols, categorical_cols = get_feature_columns()

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop"
    )

    return engineer, col_transformer
