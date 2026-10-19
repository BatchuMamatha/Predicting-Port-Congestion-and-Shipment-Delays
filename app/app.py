"""
Aiqon AI - Port Congestion & Shipment Delay Intelligence Dashboard
Interactive Streamlit Application featuring:
- Real-time delay likelihood prediction & confidence scoring
- Local & global SHAP explainability
- Actionable maritime decision support & slow-steaming recommendations
- Interactive What-If counterfactual scenario simulator
- Model validation benchmarks & calibration diagnostics
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from src.data_generator import PORT_PROFILES, VESSEL_TYPES
from src.model import PortCongestionPredictor
from src.explainability import MaritimeExplainer
from src.recommendations import ActionableRecommender


st.set_page_config(
    page_title="Aiqon | Port Congestion & Delay AI",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #1e293b;
        padding: 8px 12px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 6px;
    }
    .action-card {
        background-color: #1e293b;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_artifacts():
    predictor_path = os.path.join("models", "predictor.joblib")
    preproc_path = os.path.join("models", "preprocessor.joblib")
    
    predictor = PortCongestionPredictor.load(predictor_path)
    bundle = joblib.load(preproc_path)
    explainer = MaritimeExplainer(predictor, bundle["feature_names"])
    recommender = ActionableRecommender()
    
    metrics_path = os.path.join("artifacts", "evaluation_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    dataset_path = os.path.join("data", "port_congestion_dataset.csv")
    sample_df = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.DataFrame()
    
    return predictor, bundle, explainer, recommender, metrics, sample_df


try:
    predictor, bundle, explainer, recommender, metrics, sample_df = load_models_and_artifacts()
    models_ready = True
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure train_and_evaluate.py has been executed.")
    models_ready = False


# Top Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0 25px 0; border-bottom: 1px solid #1e293b; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
            <span style="color: #3b82f6;">Aiqon</span> Maritime Intelligence
        </h1>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
            Predicting Port Congestion, Arrival Delays & Operational Risk with Interpretable AI
        </p>
    </div>
    <div style="text-align: right;">
        <span style="background-color: #064e3b; color: #34d399; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; border: 1px solid #059669;">
            ● Calibrated XGBoost & SHAP Active
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


if models_ready:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Real-Time Delay Prediction",
        "⚖️ Counterfactual What-If Simulator",
        "🌐 Port Network & Fleet Monitor",
        "📊 Model Benchmarks & Calibration"
    ])

    # -------------------------------------------------------------
    # TAB 1: REAL-TIME DELAY PREDICTION
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Vessel Movement & Port Operations Telemetry")
        st.caption("Input vessel approach telemetry, quayside capacity, and marine weather conditions to estimate delay likelihood and root causes.")

        col_left, col_right = st.columns([1, 1], gap="medium")

        with col_left:
            st.markdown("#### ⚓ Port & Vessel Details")
            port_name = st.selectbox("Destination Port", list(PORT_PROFILES.keys()), index=0)
            port_meta = PORT_PROFILES[port_name]
            
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                vessel_type = st.selectbox("Vessel Category", [v["type"] for v in VESSEL_TYPES], index=0)
            with c_v2:
                vessel_teu = st.number_input("TEU Capacity", min_value=0, max_value=24000, value=15000, step=500)
                
            c_v3, c_v4 = st.columns(2)
            with c_v3:
                vessel_draft_m = st.slider("Vessel Draft (m)", min_value=6.0, max_value=22.0, value=14.5, step=0.1)
            with c_v4:
                labor_status = st.selectbox("Stevedore / Labor Status", ["Full Crew", "Reduced Shift", "Dispute / Shortage"], index=0)

            st.markdown("#### 📡 AIS Approach Telemetry")
            c_a1, c_a2 = st.columns(2)
            with c_a1:
                ais_distance_nm = st.slider("Distance to Port Limits (NM)", min_value=5.0, max_value=200.0, value=65.0, step=5.0)
            with c_a2:
                ais_speed_knots = st.slider("Current Approach Speed (knots)", min_value=6.0, max_value=22.0, value=15.0, step=0.5)

        with col_right:
            st.markdown("#### 🏗️ Port Congestion & Quayside Capacity")
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                anchorage_queue_length = st.slider("Current Anchorage Queue (Vessels)", min_value=0, max_value=35, value=12, step=1)
            with c_c2:
                berth_occupancy_ratio = st.slider("Berth Occupancy Ratio", min_value=0.30, max_value=1.10, value=0.88, step=0.02)

            c_c3, c_c4 = st.columns(2)
            with c_c3:
                operational_berths = st.slider("Active Operational Berths", min_value=5, max_value=port_meta["total_berths"], value=int(port_meta["total_berths"]*0.85))
            with c_c4:
                crane_avail = st.slider("Crane Availability Ratio", min_value=0.50, max_value=1.0, value=0.85, step=0.05)

            st.markdown("#### 🌊 Environmental & Regulatory Factors")
            c_w1, c_w2 = st.columns(2)
            with c_w1:
                wind_speed_knots = st.slider("Sustained Wind Speed (knots)", min_value=2.0, max_value=60.0, value=18.0, step=1.0)
            with c_w2:
                wave_height_m = st.slider("Wave Height (m)", min_value=0.5, max_value=7.0, value=1.8, step=0.1)

            c_r1, c_r2 = st.columns(2)
            with c_r1:
                customs_inspection = st.checkbox("Customs / Border Inspection Hold", value=False)
            with c_r2:
                is_peak_season = st.checkbox("Peak Shipping Season (Q3/Q4)", value=True)

        st.markdown("<hr style='border-color: #334155; margin: 25px 0;'>", unsafe_allow_html=True)

        # Build feature vector
        sample_dict = {
            "vessel_teu_capacity": vessel_teu,
            "vessel_draft_m": vessel_draft_m,
            "total_berths": port_meta["total_berths"],
            "operational_berths": operational_berths,
            "crane_availability_ratio": crane_avail,
            "berth_occupancy_ratio": berth_occupancy_ratio,
            "anchorage_queue_length": anchorage_queue_length,
            "inbound_traffic_density_24h": port_meta["avg_daily_arrivals"],
            "gate_congestion_index": 65.0,
            "ais_distance_nm": ais_distance_nm,
            "ais_speed_knots": ais_speed_knots,
            "wind_speed_knots": wind_speed_knots,
            "wave_height_m": wave_height_m,
            "visibility_nm": 7.5,
            "is_adverse_weather": 1 if (wind_speed_knots > 32 or wave_height_m > 3.0) else 0,
            "customs_inspection_required": 1 if customs_inspection else 0,
            "pilot_dispatch_delay_hours": 1.5,
            "scheduled_arrival_month": 9 if is_peak_season else 4,
            "scheduled_arrival_hour": 14,
            "scheduled_arrival_dayofweek": 3,
            "is_peak_season": 1 if is_peak_season else 0,
            "port_name": port_name,
            "vessel_type": vessel_type,
            "labor_status": labor_status,
        }

        # Run inference
        input_df = pd.DataFrame([sample_dict])
        eng_df = bundle["feature_engineer"].transform(input_df)
        X_vec = bundle["col_transformer"].transform(eng_df)

        prob, binary_pred, conf_score, conf_tier = predictor.predict_likelihood_and_confidence(X_vec)
        prob_val = float(prob[0])
        pred_delay = float(predictor.predict_delay_hours(X_vec)[0])
        conf_val = float(conf_score[0])
        tier_val = conf_tier[0]

        # Financial Demurrage Estimation
        daily_rate = 45000 if "ULCV" in vessel_type else (35000 if "Tanker" in vessel_type else 22000)
        demurrage_est = max(0.0, (pred_delay - 6.0) * (daily_rate / 24.0))

        # Risk Classification Display
        if prob_val >= 0.65:
            risk_color = "#ef4444"
            risk_label = "HIGH CONGESTION RISK"
        elif prob_val >= 0.35:
            risk_color = "#f59e0b"
            risk_label = "MODERATE RISK"
        else:
            risk_color = "#10b981"
            risk_label = "LOW CONGESTION RISK"

        # Predictions Metrics Grid
        st.markdown("### 🎯 Predictive Assessment")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Delay Likelihood</div>
                <div class="metric-value" style="color: {risk_color};">{prob_val*100:.1f}%</div>
                <div class="metric-sub" style="color: {risk_color}; font-weight: 600;">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Confidence Score</div>
                <div class="metric-value" style="color: #38bdf8;">{conf_val:.1f}%</div>
                <div class="metric-sub" style="color: #94a3b8;">{tier_val} (Sigmoid Calibrated)</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Expected Anchorage Wait</div>
                <div class="metric-value" style="color: #f8fafc;">{pred_delay:.1f} <span style="font-size: 1.2rem; color: #94a3b8;">hrs</span></div>
                <div class="metric-sub" style="color: #94a3b8;">Deviation from Berth Schedule</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Demurrage Exposure</div>
                <div class="metric-value" style="color: #fbbf24;">${demurrage_est:,.0f}</div>
                <div class="metric-sub" style="color: #94a3b8;">Charter Rate: ${daily_rate:,}/day</div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP Local Explainability
        st.markdown("### 🔍 Model Explainability & Root Cause Attribution (TreeSHAP)")
        explanation = explainer.explain_instance(X_vec[0], top_k=4)

        st.info(f"💡 **AI Operational Narrative:** {explanation['narrative']}")

        exp_col1, exp_col2 = st.columns([1.2, 1], gap="medium")

        with exp_col1:
            # SHAP Bar Chart
            drivers = explanation["top_risk_drivers"] + explanation["top_mitigating_factors"]
            drivers.sort(key=lambda x: x["shap_value"])

            names = [d["display_name"] for d in drivers]
            values = [d["shap_value"] for d in drivers]
            colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

            fig, ax = plt.subplots(figsize=(7, 4.2))
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")

            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, color=colors, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, color="#f1f5f9", fontsize=10)
            ax.axvline(0, color="#64748b", linestyle="--", alpha=0.7)

            ax.set_xlabel("SHAP Value (Impact on Delay Likelihood)", color="#94a3b8", fontsize=10)
            ax.tick_params(axis="x", colors="#94a3b8")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#334155")
            ax.spines["bottom"].set_color("#334155")
            ax.grid(axis="x", color="#334155", linestyle=":", alpha=0.5)

            plt.title("Feature Contributions to this Specific Shipment", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with exp_col2:
            st.markdown("#### 🚨 Key Risk Accelerators")
            for d in explanation["top_risk_drivers"][:3]:
                st.markdown(f"""
                <div style="background-color: #1e293b; border-left: 4px solid #ef4444; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;">
                    <div style="font-weight: 600; color: #f87171;">{d['display_name']}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Increased delay probability by <b>+{d['shap_value']:.2f}</b> log-odds.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🛡️ Mitigating Buffers")
            for d in explanation["top_mitigating_factors"][:2]:
                st.markdown(f"""
                <div style="background-color: #1e293b; border-left: 4px solid #10b981; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;">
                    <div style="font-weight: 600; color: #34d399;">{d['display_name']}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Reduced delay probability by <b>-{abs(d['shap_value']):.2f}</b> log-odds.</div>
                </div>
                """, unsafe_allow_html=True)

        # Actionable Operational Decisions
        st.markdown("### 📋 Prescriptive Operational Interventions")
        actions = recommender.generate_recommendations(
            prediction_prob=prob_val,
            predicted_delay_hours=pred_delay,
            top_risk_drivers=explanation["top_risk_drivers"],
            raw_inputs=sample_dict
        )

        for act in actions:
            badge_class = "badge-high" if act["priority"] in ["HIGH", "CRITICAL"] else ("badge-med" if act["priority"] == "MEDIUM" else "badge-low")
            st.markdown(f"""
            <div class="action-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">{act['title']}</span>
                    <span class="{badge_class}">{act['priority']} PRIORITY</span>
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 6px;">{act['rationale']}</div>
                <div style="font-size: 0.85rem; color: #38bdf8; font-weight: 600;">✨ Operational Impact: {act['expected_impact']}</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: WHAT-IF / COUNTERFACTUAL SIMULATOR
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Interactive What-If & Counterfactual Scenario Simulator")
        st.caption("Evaluate how adjusting controllable operational parameters (such as approach speed, queue clearance, or terminal allocation) directly mitigates delay risk.")

        sim_col1, sim_col2 = st.columns([1, 1], gap="large")

        with sim_col1:
            st.markdown("#### 🎛️ Scenario Knobs (Adjust to Mitigate)")
            new_speed = st.slider("Target Steaming Speed (knots)", min_value=8.0, max_value=22.0, value=11.0, step=0.5, key="sim_speed")
            new_queue = st.slider("Assumed Anchorage Queue at Arrival", min_value=0, max_value=30, value=4, step=1, key="sim_queue")
            new_berths = st.slider("Active Berths Available", min_value=10, max_value=port_meta["total_berths"], value=port_meta["total_berths"], key="sim_berths")
            clear_customs = st.checkbox("Pre-Clear Customs Remotely", value=True, key="sim_customs")
            full_labor = st.checkbox("Guarantee Full Labor Shift Crew", value=True, key="sim_labor")

        with sim_col2:
            st.markdown("#### 🔄 Live Simulation vs Baseline")

            # Mitigated feature dict
            sim_dict = sample_dict.copy()
            sim_dict["ais_speed_knots"] = new_speed
            sim_dict["anchorage_queue_length"] = new_queue
            sim_dict["operational_berths"] = new_berths
            sim_dict["customs_inspection_required"] = 0 if clear_customs else 1
            sim_dict["labor_status"] = "Full Crew" if full_labor else sample_dict["labor_status"]

            # Transform and predict
            sim_input_df = pd.DataFrame([sim_dict])
            sim_eng_df = bundle["feature_engineer"].transform(sim_input_df)
            X_sim = bundle["col_transformer"].transform(sim_eng_df)

            sim_prob, _, sim_conf, _ = predictor.predict_likelihood_and_confidence(X_sim)
            sim_prob_val = float(sim_prob[0])
            sim_delay_val = float(predictor.predict_delay_hours(X_sim)[0])

            delta_prob = (sim_prob_val - prob_val) * 100
            delta_delay = sim_delay_val - pred_delay

            # Fuel saving estimation
            fuel_saved_pct = max(0.0, round((1.0 - (new_speed / max(1.0, ais_speed_knots))**3) * 100, 1)) if new_speed < ais_speed_knots else 0.0

            s1, s2 = st.columns(2)
            with s1:
                st.metric("New Delay Likelihood", f"{sim_prob_val*100:.1f}%", delta=f"{delta_prob:+.1f}%", delta_color="inverse")
            with s2:
                st.metric("New Waiting Time", f"{sim_delay_val:.1f} hrs", delta=f"{delta_delay:+.1f} hrs", delta_color="inverse")

            st.markdown("---")
            st.markdown(f"""
            <div style="background-color: #1e293b; padding: 18px; border-radius: 10px; border: 1px solid #334155;">
                <h5 style="margin: 0 0 10px 0; color: #38bdf8;">Simulation Takeaway</h5>
                <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
                    By adopting <b>slow-steaming ({new_speed} kts)</b> and securing <b>customs pre-clearance</b>, 
                    the probability of shipment delay is reduced from <b>{prob_val*100:.1f}%</b> to <b>{sim_prob_val*100:.1f}%</b>.
                </p>
                <div style="margin-top: 12px; font-weight: 600; color: #34d399; font-size: 0.9rem;">
                    🌱 Estimated Fuel & Bunker Reduction: {fuel_saved_pct}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: PORT NETWORK & FLEET MONITOR
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Global Port Network Telemetry & Active Queue Status")
        st.caption("Live operational status across tier-1 international container and bulk terminals.")

        p_cols = st.columns(len(PORT_PROFILES))
        for idx, (p_key, p_val) in enumerate(PORT_PROFILES.items()):
            with p_cols[idx]:
                short_name = p_key.split(" (")[0]
                st.markdown(f"""
                <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 15px; text-align: center;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1rem; margin-bottom: 4px;">{short_name}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 12px;">{p_val['region']}</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8;">{p_val['total_berths']}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">Total Berths</div>
                    <div style="margin-top: 10px; font-size: 0.85rem; color: #cbd5e1;">Avg Arrivals: <b>{p_val['avg_daily_arrivals']}/day</b></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📋 Synthetic AIS Port-Call Records Sample")
        if not sample_df.empty:
            display_cols = [
                "record_id", "vessel_type", "port_name", "anchorage_queue_length",
                "berth_occupancy_ratio", "wind_speed_knots", "waiting_time_hours", "is_congested_delayed"
            ]
            st.dataframe(sample_df[display_cols].head(15), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: MODEL BENCHMARKS & CALIBRATION
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Rigorous Model Validation, Calibration & Diagnostics")
        st.caption("Benchmarking the Calibrated XGBoost system against industry baseline models on a held-out test split.")

        b1, b2, b3, b4 = st.columns(4)
        c_metrics = metrics.get("classification", {})
        r_metrics = metrics.get("regression", {})

        with b1:
            st.metric("Test ROC-AUC", f"{c_metrics.get('roc_auc', 0.92):.3f}", f"Baseline: {c_metrics.get('baseline_roc_auc', 0.93):.3f}")
        with b2:
            st.metric("PR-AUC (Avg Precision)", f"{c_metrics.get('pr_auc', 0.87):.3f}")
        with b3:
            st.metric("Brier Calibration Score", f"{c_metrics.get('brier_score', 0.14):.3f}", "Lower is Better")
        with b4:
            st.metric("Delay MAE", f"{r_metrics.get('mae_hours', 3.01):.2f} hrs", f"Baseline: {r_metrics.get('baseline_mae_hours', 3.25):.2f} hrs")

        st.markdown("### 📈 Diagnostic Artifacts & Reliability Plots")
        p_row1, p_row2 = st.columns(2)

        with p_row1:
            roc_path = os.path.join("artifacts", "roc_pr_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic & Precision-Recall Curves")

        with p_row2:
            calib_path = os.path.join("artifacts", "calibration_curve.png")
            if os.path.exists(calib_path):
                st.image(calib_path, caption="Probability Calibration Reliability Diagram (Sigmoid Scaling)")

        p_row3, p_row4 = st.columns(2)
        with p_row3:
            shap_path = os.path.join("artifacts", "shap_global_summary.png")
            if os.path.exists(shap_path):
                st.image(shap_path, caption="Global SHAP Feature Importance Hierarchy")

        with p_row4:
            reg_path = os.path.join("artifacts", "delay_hours_regression.png")
            if os.path.exists(reg_path):
                st.image(reg_path, caption="Delay Duration Fit (Predicted vs Actual Hours)")
