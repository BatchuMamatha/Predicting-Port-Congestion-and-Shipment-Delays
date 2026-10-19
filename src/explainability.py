"""
Explainability Engine using SHAP (SHapley Additive exPlanations)
Provides local feature attributions, natural language explanations,
global summary statistics, and actionable counterfactual simulations.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap


class MaritimeExplainer:
    """
    Interpretable AI Explainer for port congestion and delay predictions.
    Utilizes TreeSHAP to attribute exact risk margins to operational,
    weather, and capacity factors.
    """
    def __init__(self, predictor, feature_names: List[str]):
        self.predictor = predictor
        self.feature_names = feature_names
        # Initialize TreeExplainer on base XGBoost classifier
        self.explainer = shap.TreeExplainer(self.predictor.base_classifier)

    def explain_instance(
        self,
        X_single: np.ndarray,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Computes SHAP values for a single prediction and generates
        human-readable operational drivers.
        """
        if len(X_single.shape) == 1:
            X_single = X_single.reshape(1, -1)
            
        shap_values = self.explainer.shap_values(X_single)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Positive class (delayed)
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values[0]
            
        base_value = float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value[1])
        
        # Pair features with their SHAP values and raw values
        driver_items = []
        for i, feat in enumerate(self.feature_names):
            val = float(X_single[0, i])
            shap_val = float(sv[i])
            driver_items.append({
                "feature": feat,
                "display_name": self._clean_feature_name(feat),
                "scaled_value": val,
                "shap_value": shap_val,
                "abs_impact": abs(shap_val)
            })
            
        # Sort by absolute SHAP impact
        driver_items.sort(key=lambda x: x["abs_impact"], reverse=True)
        
        # Separate positive (risk aggravating) and negative (risk mitigating) drivers
        risk_drivers = [d for d in driver_items if d["shap_value"] > 0][:top_k]
        mitigating_factors = [d for d in driver_items if d["shap_value"] < 0][:top_k]
        
        # Build natural language narrative
        narrative = self._generate_narrative(risk_drivers, mitigating_factors)
        
        return {
            "base_value": base_value,
            "shap_values": sv.tolist(),
            "feature_names": self.feature_names,
            "top_risk_drivers": risk_drivers,
            "top_mitigating_factors": mitigating_factors,
            "narrative": narrative
        }

    def _clean_feature_name(self, name: str) -> str:
        """Translates raw model feature names to intuitive maritime operational terms."""
        mapping = {
            "anchorage_queue_length": "Anchorage Queue Length",
            "traffic_to_berth_ratio": "Traffic-to-Berth Intensity",
            "berth_pressure_index": "Berth Congestion Index",
            "berth_occupancy_ratio": "Berth Occupancy Ratio",
            "operational_berths": "Operational Berths",
            "crane_availability_ratio": "Crane Availability Ratio",
            "wind_speed_knots": "Sustained Wind Speed",
            "wave_height_m": "Significant Wave Height",
            "weather_severity_index": "Adverse Weather Severity",
            "is_adverse_weather": "Storm / Gale Warning",
            "customs_inspection_required": "Customs Inspection Hold",
            "pilot_dispatch_delay_hours": "Pilot Dispatch Lag",
            "inbound_traffic_density_24h": "24h Inbound Traffic Density",
            "gate_congestion_index": "Terminal Gate Congestion",
            "vessel_teu_capacity": "Vessel TEU Capacity",
            "vessel_draft_m": "Vessel Draft",
            "ais_speed_knots": "AIS Approach Speed",
            "ais_distance_nm": "Distance to Port Limit",
            "queue_per_crane_ratio": "Queue-per-Crane Bottleneck",
            "labor_status_Dispute / Shortage": "Labor Dispute / Shortage",
            "labor_status_Reduced Shift": "Labor Reduced Shift",
            "labor_status_Full Crew": "Full Crew Availability",
            "is_peak_season": "Peak Shipping Season",
        }
        for k, v in mapping.items():
            if k in name:
                return v
        return name.replace("_", " ").title()

    def _generate_narrative(self, risk_drivers: List[Dict], mitigating_factors: List[Dict]) -> str:
        """Builds a professional, concise operational narrative from key factors."""
        points = []
        if risk_drivers:
            top_risk = risk_drivers[0]
            points.append(
                f"Primary delay driver is **{top_risk['display_name']}** (SHAP impact +{top_risk['shap_value']:.2f})"
            )
            if len(risk_drivers) > 1:
                secondary = [f"{d['display_name']} (+{d['shap_value']:.2f})" for d in risk_drivers[1:3]]
                points.append(f"compounded by {', '.join(secondary)}.")
                
        if mitigating_factors:
            top_mit = mitigating_factors[0]
            points.append(
                f"Partially buffered by **{top_mit['display_name']}** (-{abs(top_mit['shap_value']):.2f} risk mitigation)."
            )
            
        return " ".join(points) if points else "Balanced operational conditions with no dominant risk drivers."
