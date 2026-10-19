"""
Actionable Decision Support & Maritime Operational Recommendations Engine
Translates predictive likelihoods and SHAP drivers into cost-saving,
emission-reducing operational interventions.
"""

from typing import Dict, Any, List


class ActionableRecommender:
    """
    Maritime Operational Decision Engine.
    Converts ML predictions into prioritized, concrete actions:
    - Virtual Arrival / Slow Steaming (Bunker Fuel & Emissions Savings)
    - Terminal / Berth Reallocation
    - Customs & Regulatory Pre-Clearance
    - Tidal Window Synchronization
    """
    def __init__(self):
        pass

    def generate_recommendations(
        self,
        prediction_prob: float,
        predicted_delay_hours: float,
        top_risk_drivers: List[Dict[str, Any]],
        raw_inputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes model output and domain drivers into prioritized action items.
        """
        actions = []
        
        # 1. Slow Steaming / Virtual Arrival Recommendation
        current_speed = float(raw_inputs.get("ais_speed_knots", 14.0))
        dist_nm = float(raw_inputs.get("ais_distance_nm", 50.0))
        
        if predicted_delay_hours >= 6.0 and dist_nm >= 20.0 and current_speed > 11.0:
            recommended_speed = max(9.5, round(current_speed * 0.75, 1))
            hours_saved_at_anchor = round(predicted_delay_hours * 0.55, 1)
            # Fuel saving estimation (cubic law: fuel ~ speed^3)
            fuel_reduction_pct = round((1 - (recommended_speed / current_speed)**3) * 100, 1)
            
            actions.append({
                "category": "Virtual Arrival / Speed Optimization",
                "priority": "HIGH",
                "title": f"Initiate Slow-Steaming: Throttle from {current_speed} to {recommended_speed} kts",
                "rationale": (
                    f"With a predicted {predicted_delay_hours:.1f}h anchorage delay, arriving earlier will only increase idle time in congested roads. "
                    f"Absorbing delay at sea reduces anchorage congestion and cuts voyage bunker fuel consumption by ~{fuel_reduction_pct}%."
                ),
                "expected_impact": f"Replaces ~{hours_saved_at_anchor}h of anchorage idling with smooth berth synchronization."
            })

        # 2. Anchorage & Queue Bottlenecks
        queue_len = int(raw_inputs.get("anchorage_queue_length", 0))
        berth_occ = float(raw_inputs.get("berth_occupancy_ratio", 0.7))
        
        if queue_len >= 8 or berth_occ >= 0.88:
            actions.append({
                "category": "Terminal & Berth Allocation",
                "priority": "HIGH",
                "title": "Request Off-Peak Berth Window or Secondary Terminal Allocation",
                "rationale": (
                    f"Port queue is currently {queue_len} vessels with {berth_occ*100:.0f}% berth occupancy. "
                    f"Liaise with Port Authority (VTS) to evaluate secondary terminal berths or flexible nighttime discharge windows."
                ),
                "expected_impact": "Reduces queue wait by an estimated 4.0 - 8.5 hours."
            })

        # 3. Weather Interventions
        wind_speed = float(raw_inputs.get("wind_speed_knots", 10.0))
        wave_height = float(raw_inputs.get("wave_height_m", 1.0))
        
        if wind_speed >= 30.0 or wave_height >= 2.5:
            actions.append({
                "category": "Adverse Weather Protocol",
                "priority": "CRITICAL" if wind_speed > 35 else "MEDIUM",
                "title": "Activate Heavy Weather Berthing & Crane Contingency Protocol",
                "rationale": (
                    f"Sustained winds of {wind_speed:.1f} kts and {wave_height:.1f}m waves exceed safe STS crane handling thresholds (typically 35 kts). "
                    f"Quayside crane operations will face intermittent stops."
                ),
                "expected_impact": "Secure extra tug assistance in advance and pre-schedule pilot boarding for sheltered fairway approaches."
            })

        # 4. Customs & Regulatory Pre-Clearance
        customs_req = raw_inputs.get("customs_inspection_required", 0)
        if customs_req == 1 or any("customs" in d.get("feature", "").lower() for d in top_risk_drivers):
            actions.append({
                "category": "Regulatory & Documentation",
                "priority": "MEDIUM",
                "title": "Initiate Offshore Electronic Customs Pre-Clearance (EDI)",
                "rationale": (
                    "Customs inspection hold identified as a critical bottleneck driver. Submitting electronic manifest and automated clearance documents "
                    "prior to arrival allows immediate green-channel processing upon berth."
                ),
                "expected_impact": "Eliminates shoreside documentation hold (estimated saving: 6.0 - 14.0 hours)."
            })

        # 5. Labor / Shift Contingency
        labor_status = raw_inputs.get("labor_status", "Full Crew")
        if labor_status != "Full Crew":
            actions.append({
                "category": "Stevedore Resource Management",
                "priority": "HIGH",
                "title": f"Coordinate Advance Stevedore Gang Allocation ({labor_status})",
                "rationale": (
                    f"Current terminal status is '{labor_status}', reducing hourly TEU discharge rates. "
                    f"Pre-contract specialized crane gang crews for immediate vessel turnaround."
                ),
                "expected_impact": "Maintains crane discharge rates at near-normal capacity."
            })
            
        # Default reassuring action if risk is low
        if not actions and prediction_prob < 0.35:
            actions.append({
                "category": "Standard Operations",
                "priority": "LOW",
                "title": "Maintain Current Approach & Standard Berth Reservation",
                "rationale": "Conditions are optimal. Port capacity, queue status, and weather present minimal delay risk.",
                "expected_impact": "On-time arrival and expected direct berthing."
            })

        return actions
