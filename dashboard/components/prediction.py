"""
Dashboard Page 5: Single Transaction Prediction
Interactive risk scoring simulator with preset legitimate and fraudulent profiles.
"""

import streamlit as st
import plotly.graph_objects as go
from src.models.predict import predict_single_transaction

def render_transaction_prediction():
    st.markdown("## 🛡️ Interactive Transaction Risk Evaluator")
    st.markdown("Test single credit card transactions in real time to simulate point-of-sale risk evaluation.")

    # Preset Quick Load Buttons
    st.subheader("1. Select a Preset or Configure Custom Values")
    col_pre1, col_pre2, col_pre3 = st.columns(3)

    if "tx_inputs" not in st.session_state:
        # Default to typical legitimate values
        st.session_state.tx_inputs = {
            "Time": 42500.0,
            "Amount": 45.50,
            "V14": 0.12,
            "V10": -0.05,
            "V4": -0.25,
            "V12": 0.15,
            "V17": 0.08,
            "V11": -0.10,
            "V16": 0.05
        }

    with col_pre1:
        if st.button("🟢 Load Typical Legitimate Profile", use_container_width=True):
            st.session_state.tx_inputs = {
                "Time": 50000.0,
                "Amount": 38.25,
                "V14": 0.25,
                "V10": 0.05,
                "V4": -0.18,
                "V12": 0.10,
                "V17": -0.02,
                "V11": -0.30,
                "V16": 0.12
            }
            st.rerun()

    with col_pre2:
        if st.button("🔴 Load Confirmed Fraud Profile", use_container_width=True):
            st.session_state.tx_inputs = {
                "Time": 41200.0,
                "Amount": 312.00,
                "V14": -6.85,
                "V10": -4.20,
                "V4": 4.15,
                "V12": -5.90,
                "V17": -5.10,
                "V11": 3.80,
                "V16": -2.40
            }
            st.rerun()

    with col_pre3:
        if st.button("🟡 Load Borderline / Suspicious Profile", use_container_width=True):
            st.session_state.tx_inputs = {
                "Time": 15000.0,
                "Amount": 850.00,
                "V14": -2.10,
                "V10": -1.60,
                "V4": 1.80,
                "V12": -1.90,
                "V17": -1.50,
                "V11": 1.40,
                "V16": -0.90
            }
            st.rerun()

    st.markdown("---")

    # Input Form
    st.subheader("2. Transaction Parameters")
    with st.form("prediction_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            amount_val = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=float(st.session_state.tx_inputs["Amount"]), step=5.0)
            time_val = st.number_input("Transaction Time (Seconds)", min_value=0.0, max_value=200000.0, value=float(st.session_state.tx_inputs["Time"]), step=100.0)
            v14_val = st.slider("V14 (Primary Fraud Discriminator)", min_value=-15.0, max_value=5.0, value=float(st.session_state.tx_inputs["V14"]), step=0.1)
            v10_val = st.slider("V10 (Secondary Fraud Indicator)", min_value=-15.0, max_value=5.0, value=float(st.session_state.tx_inputs["V10"]), step=0.1)
        with col_f2:
            v4_val = st.slider("V4 (Anomaly Indicator)", min_value=-5.0, max_value=12.0, value=float(st.session_state.tx_inputs["V4"]), step=0.1)
            v12_val = st.slider("V12 (PCA Latent Factor)", min_value=-15.0, max_value=5.0, value=float(st.session_state.tx_inputs["V12"]), step=0.1)
            v17_val = st.slider("V17 (PCA Latent Factor)", min_value=-15.0, max_value=5.0, value=float(st.session_state.tx_inputs["V17"]), step=0.1)
            v11_val = st.slider("V11 (PCA Latent Factor)", min_value=-5.0, max_value=10.0, value=float(st.session_state.tx_inputs["V11"]), step=0.1)

        submit_btn = st.form_submit_button("⚡ Run Real-Time Risk Evaluation", type="primary", use_container_width=True)

    # Assemble and Predict
    tx_payload = {
        "transaction_id": "tx_manual_test",
        "Time": time_val,
        "Amount": amount_val,
        "V14": v14_val,
        "V10": v10_val,
        "V4": v4_val,
        "V12": v12_val,
        "V17": v17_val,
        "V11": v11_val,
        "V16": float(st.session_state.tx_inputs.get("V16", 0.0))
    }
    # Fill any remaining V features with neutral 0.0
    for i in range(1, 29):
        if f"V{i}" not in tx_payload:
            tx_payload[f"V{i}"] = 0.0

    result = predict_single_transaction(tx_payload)

    st.markdown("---")
    st.subheader("3. Risk Evaluation Results")

    res_col1, res_col2 = st.columns([2, 3])

    with res_col1:
        prob = result["fraud_probability"]
        risk = result["risk_level"]
        rec = result["recommendation"]
        
        # Color mapping
        color_map = {
            "LOW": "#10B981",
            "MEDIUM": "#F59E0B",
            "HIGH": "#F97316",
            "CRITICAL": "#EF4444"
        }
        badge_color = color_map.get(risk, "#6B7280")

        st.markdown(
            f"""
            <div style="background-color: {badge_color}20; border-left: 6px solid {badge_color}; padding: 18px; border-radius: 8px;">
                <h3 style="color: {badge_color}; margin: 0;">Risk Tier: {risk}</h3>
                <h4 style="margin: 8px 0;">Decision: {'🚨 SUSPICIOUS / FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}</h4>
                <p style="margin: 0; font-size: 15px;"><b>Recommended Action:</b> <code>{rec}</code></p>
                <p style="margin: 4px 0 0 0; font-size: 14px;">Fraud Probability: <b>{prob:.2%}</b> (Threshold: {result['threshold_used']})</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col2:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Estimated Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": badge_color},
                "steps": [
                    {"range": [0, 30], "color": "#E2E8F0"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 85], "color": "#FFEDD5"},
                    {"range": [85, 100], "color": "#FEE2E2"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": result["threshold_used"] * 100
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
