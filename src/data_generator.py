"""
Port Congestion & AIS Vessel Activity Dataset Generator
Simulates realistic maritime logistics data based on queueing dynamics (M/M/c),
weather conditions, berth constraints, labor efficiency, and temporal patterns.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Port operational profiles with realistic berth counts, mean service rates, and climate traits
PORT_PROFILES: Dict[str, Dict[str, Any]] = {
    "Port of Singapore (SGSIN)": {
        "region": "Southeast Asia",
        "total_berths": 65,
        "base_service_rate_per_berth": 0.085,  # vessels cleared per berth hour (~11.7 hrs per vessel)
        "avg_daily_arrivals": 55,
        "weather_risk_factor": 0.4,  # monsoons/tropical squalls
        "max_draft_m": 18.0,
    },
    "Port of Rotterdam (NLRTM)": {
        "region": "Northern Europe",
        "total_berths": 48,
        "base_service_rate_per_berth": 0.075,  # ~13.3 hrs per vessel
        "avg_daily_arrivals": 40,
        "weather_risk_factor": 0.6,  # North Sea gales / fog
        "max_draft_m": 24.0,
    },
    "Port of Los Angeles (USLAX)": {
        "region": "North America West Coast",
        "total_berths": 32,
        "base_service_rate_per_berth": 0.060,  # ~16.6 hrs per vessel (chassis/labor bottlenecks)
        "avg_daily_arrivals": 28,
        "weather_risk_factor": 0.3,  # Santa Ana winds / swell
        "max_draft_m": 16.0,
    },
    "Port of Shanghai (CNSHA)": {
        "region": "East Asia",
        "total_berths": 80,
        "base_service_rate_per_berth": 0.090,  # high crane throughput
        "avg_daily_arrivals": 75,
        "weather_risk_factor": 0.7,  # typhoons / dense Yangtze fog
        "max_draft_m": 17.5,
    },
    "Jawaharlal Nehru Port (INNSA)": {
        "region": "South Asia",
        "total_berths": 22,
        "base_service_rate_per_berth": 0.065,  # ~15.3 hrs per vessel
        "avg_daily_arrivals": 20,
        "weather_risk_factor": 0.65,  # Southwest Monsoon disruption
        "max_draft_m": 15.0,
    },
}

VESSEL_TYPES = [
    {"type": "Ultra Large Container (ULCV)", "weight": 0.25, "mean_teu": 18000, "base_time_h": 28.0, "mean_draft": 15.5},
    {"type": "Panamax Container", "weight": 0.30, "mean_teu": 5000, "base_time_h": 16.0, "mean_draft": 12.0},
    {"type": "Capesize Bulk Carrier", "weight": 0.15, "mean_teu": 0, "base_time_h": 36.0, "mean_draft": 16.0},
    {"type": "Handysize Bulk", "weight": 0.15, "mean_teu": 0, "base_time_h": 18.0, "mean_draft": 10.0},
    {"type": "Suezmax Crude Tanker", "weight": 0.15, "mean_teu": 0, "base_time_h": 24.0, "mean_draft": 14.5},
]


def generate_synthetic_port_dataset(
    n_samples: int = 2500,
    random_seed: int = 42,
    delay_threshold_hours: float = 12.0
) -> pd.DataFrame:
    """
    Generate synthetic port congestion and vessel arrival data.
    
    Parameters:
    -----------
    n_samples : int
        Number of port-call records to synthesize.
    random_seed : int
        Random seed for reproducibility.
    delay_threshold_hours : float
        Threshold above which a vessel is considered 'Congested / Delayed'.
        
    Returns:
    --------
    pd.DataFrame with AIS, port operational, weather, and congestion delay indicators.
    """
    np.random.seed(random_seed)
    
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    records = []
    
    port_names = list(PORT_PROFILES.keys())
    port_weights = [0.25, 0.22, 0.18, 0.23, 0.12]
    
    v_types = [v["type"] for v in VESSEL_TYPES]
    v_weights = [v["weight"] for v in VESSEL_TYPES]
    
    for i in range(n_samples):
        record_id = f"PC-2025-{i+10001:06d}"
        
        # 1. Port Selection & Characteristics
        port_name = np.random.choice(port_names, p=port_weights)
        port_info = PORT_PROFILES[port_name]
        total_berths = port_info["total_berths"]
        
        # 2. Vessel Characteristics
        v_idx = np.random.choice(len(VESSEL_TYPES), p=v_weights)
        v_spec = VESSEL_TYPES[v_idx]
        vessel_type = v_spec["type"]
        vessel_teu = int(max(0, np.random.normal(v_spec["mean_teu"], v_spec["mean_teu"] * 0.18)))
        vessel_draft_m = float(np.clip(np.random.normal(v_spec["mean_draft"], 1.2), 6.0, 22.0))
        vessel_id = f"IMO{np.random.randint(9000000, 9999999)}"
        
        # 3. Temporal Factors
        time_offset_days = np.random.uniform(0, 360)
        arrival_dt = start_date + timedelta(days=time_offset_days, hours=np.random.uniform(0, 24))
        arrival_month = arrival_dt.month
        arrival_hour = arrival_dt.hour
        arrival_dayofweek = arrival_dt.weekday()  # 0=Monday, 6=Sunday
        
        # Seasonality: peak shipping season (August to November)
        is_peak_season = 1 if arrival_month in [8, 9, 10, 11] else 0
        seasonal_traffic_multiplier = 1.35 if is_peak_season else (0.85 if arrival_month in [1, 2] else 1.0)
        
        # 4. AIS Tracking & Approach Dynamics
        # Distance to port when monitored (5 to 150 nautical miles)
        ais_distance_nm = float(np.round(np.random.uniform(5.0, 150.0), 1))
        # Current vessel speed (cruising speed vs slow-steaming)
        ais_speed_knots = float(np.round(np.random.normal(14.0, 2.5), 1))
        ais_speed_knots = max(6.0, min(24.0, ais_speed_knots))
        
        # 5. Port Capacity & Operational State
        crane_availability_ratio = float(np.clip(np.random.beta(8, 2), 0.50, 1.0))
        operational_berths = int(np.clip(np.round(total_berths * crane_availability_ratio), 8, total_berths))
        
        # Labor shift availability: 0 = Normal/Full, 1 = Reduced/Holiday, 2 = Severe Shortage/Dispute
        labor_status_draw = np.random.uniform(0, 1)
        if labor_status_draw > 0.90:
            labor_status = "Dispute / Shortage"
            labor_multiplier = 1.70
        elif labor_status_draw > 0.70:
            labor_status = "Reduced Shift"
            labor_multiplier = 1.25
        else:
            labor_status = "Full Crew"
            labor_multiplier = 1.0
            
        # Berth occupancy ratio in recent 12 hours (0.4 to 1.05)
        base_occupancy = np.random.beta(6, 3) * seasonal_traffic_multiplier
        berth_occupancy_ratio = float(np.clip(base_occupancy, 0.40, 1.10))
        
        # Anchorage queue: number of vessels currently waiting in roads
        queue_lambda = (total_berths * 0.35) * seasonal_traffic_multiplier * (1.2 if arrival_dayofweek in [0, 4] else 0.9)
        anchorage_queue_length = int(np.random.poisson(queue_lambda))
        
        # 24h inbound traffic density (vessels expected in the vessel traffic separation scheme)
        inbound_traffic_density = int(np.random.poisson(port_info["avg_daily_arrivals"] * seasonal_traffic_multiplier))
        
        # Terminal gate congestion index (0 to 100)
        gate_congestion_index = float(np.clip(np.random.normal(55 + (35 * (berth_occupancy_ratio - 0.7)), 15), 10.0, 100.0))
        
        # 6. Environmental / Weather Conditions
        # Regional weather risk modulation
        wind_mean = 12.0 + (port_info["weather_risk_factor"] * 10.0)
        wind_speed_knots = float(np.clip(np.random.weibull(2.0) * wind_mean, 2.0, 60.0))
        wave_height_m = float(np.clip((wind_speed_knots / 10.0) * np.random.uniform(0.6, 1.4), 0.5, 7.5))
        visibility_nm = float(np.clip(np.random.normal(8.0 - (port_info["weather_risk_factor"] * 2), 2.5), 0.5, 12.0))
        
        # Storm warning condition (wind > 32 knots or wave > 3.0m or visibility < 1.5nm)
        is_adverse_weather = 1 if (wind_speed_knots > 32.0 or wave_height_m > 3.0 or visibility_nm < 1.5) else 0
        
        # 7. Regulatory & Logistics Factors
        customs_inspection_required = 1 if np.random.uniform(0, 1) < 0.15 else 0
        pilot_dispatch_delay_hours = float(np.clip(np.random.exponential(0.8), 0.2, 5.0))
        if is_adverse_weather:
            pilot_dispatch_delay_hours += np.random.uniform(2.0, 6.0)
            
        # 8. Ground-Truth Delay & Congestion Physics Simulation
        # Modeled via modified M/M/c queueing approximation + weather + operational penalties
        effective_capacity = operational_berths * port_info["base_service_rate_per_berth"] * 24.0 # vessels per day
        traffic_load_ratio = inbound_traffic_density / max(1.0, effective_capacity)
        
        # Base queue delay hours
        queue_wait_base = (anchorage_queue_length / max(1, operational_berths)) * v_spec["base_time_h"] * 0.45
        
        # Weather delay factor: cranes stop if wind > 35 knots, vessel berthing restricted if wave > 2.5m
        weather_delay_hours = 0.0
        if wind_speed_knots > 35.0:
            weather_delay_hours += (wind_speed_knots - 35.0) * 0.65
        if wave_height_m > 2.8:
            weather_delay_hours += (wave_height_m - 2.8) * 3.5
        if visibility_nm < 2.0:
            weather_delay_hours += (2.0 - visibility_nm) * 2.0
            
        # Draft restriction: if vessel draft is close to port max draft, must wait for high tide (6 to 12 hrs)
        draft_clearance = port_info["max_draft_m"] - vessel_draft_m
        tidal_delay_hours = 0.0
        if draft_clearance < 1.5:
            tidal_delay_hours = np.random.uniform(4.0, 11.5)
            
        # Customs hold
        customs_delay_hours = np.random.uniform(6.0, 18.0) if customs_inspection_required else 0.0
        
        # Synthesis with stochastic log-normal noise
        total_deterministic_wait = (
            queue_wait_base * labor_multiplier +
            weather_delay_hours +
            tidal_delay_hours +
            customs_delay_hours +
            pilot_dispatch_delay_hours +
            (10.0 * max(0.0, berth_occupancy_ratio - 0.85))
        )
        
        # Stochastic variance (unforeseen equipment breakdowns, bunkering lags, priority vessels)
        stochastic_noise = np.random.lognormal(mean=0.0, sigma=0.25)
        waiting_time_hours = float(np.round(max(0.5, total_deterministic_wait * stochastic_noise), 2))
        
        # Binary target: Congestion / Delay flag
        is_delayed = 1 if waiting_time_hours >= delay_threshold_hours else 0
        
        # Demurrage cost estimation ($20k-$60k / day depending on vessel size)
        daily_charter_rate = 45000 if "ULCV" in vessel_type else (35000 if "Tanker" in vessel_type else 22000)
        demurrage_usd = float(np.round(max(0.0, (waiting_time_hours - 6.0) * (daily_charter_rate / 24.0)), 2))
        
        records.append({
            "record_id": record_id,
            "vessel_id": vessel_id,
            "vessel_type": vessel_type,
            "vessel_teu_capacity": vessel_teu,
            "vessel_draft_m": vessel_draft_m,
            "port_name": port_name,
            "port_region": port_info["region"],
            "total_berths": total_berths,
            "operational_berths": operational_berths,
            "crane_availability_ratio": np.round(crane_availability_ratio, 3),
            "berth_occupancy_ratio": np.round(berth_occupancy_ratio, 3),
            "anchorage_queue_length": anchorage_queue_length,
            "inbound_traffic_density_24h": inbound_traffic_density,
            "gate_congestion_index": np.round(gate_congestion_index, 1),
            "labor_status": labor_status,
            "ais_distance_nm": ais_distance_nm,
            "ais_speed_knots": ais_speed_knots,
            "wind_speed_knots": np.round(wind_speed_knots, 1),
            "wave_height_m": np.round(wave_height_m, 2),
            "visibility_nm": np.round(visibility_nm, 1),
            "is_adverse_weather": is_adverse_weather,
            "customs_inspection_required": customs_inspection_required,
            "pilot_dispatch_delay_hours": np.round(pilot_dispatch_delay_hours, 2),
            "scheduled_arrival_month": arrival_month,
            "scheduled_arrival_hour": arrival_hour,
            "scheduled_arrival_dayofweek": arrival_dayofweek,
            "is_peak_season": is_peak_season,
            # Targets
            "waiting_time_hours": waiting_time_hours,
            "is_congested_delayed": is_delayed,
            "demurrage_usd": demurrage_usd
        })
        
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_synthetic_port_dataset(n_samples=2500)
    print(f"Dataset generated with {len(df)} records.")
    print("Class distribution:")
    print(df["is_congested_delayed"].value_counts(normalize=True))
    print("\nDelay summary (hours):")
    print(df["waiting_time_hours"].describe())
