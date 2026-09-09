"""
Dashboard Page 3: Real-Time Fraud Detection & Alert Stream
Interactive event-driven monitoring console with live transaction streaming,
sub-millisecond model inference, probability timeline, and dynamic risk alerts.
"""

import time
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.config import DEFAULT_FRAUD_THRESHOLD, get_risk_level
from src.database.queries import (
    save_transaction,
    save_prediction,
    save_alert,
    get_recent_alerts,
    get_recent_transactions
)
from src.models.predict import predict_single_transaction
from src.data.loader import load_transaction_data

def _get_stream_sample_transactions():
    """Retrieves or caches a pool of realistic transactions for real-time streaming."""
    if "stream_tx_pool" not in st.session_state:
        df = load_transaction_data(source="auto", clean=True)
        # Keep a split of legit and fraud for controlled streaming
        st.session_state.pool_fraud = df[df["Class"] == 1].to_dict(orient="records")
        st.session_state.pool_legit = df[df["Class"] == 0].to_dict(orient="records")
        st.session_state.fraud_idx = 0
        st.session_state.legit_idx = 0
        st.session_state.stream_tx_pool = True

def _fetch_next_transaction(force_fraud: bool = False, fraud_ratio: float = 0.20) -> dict:
    """Selects the next transaction from the transaction pool."""
    _get_stream_sample_transactions()
    
    fraud_pool = st.session_state.pool_fraud
    legit_pool = st.session_state.pool_legit
    
    # Decide whether this transaction should be fraud
    is_fraud = force_fraud or (np.random.random() < fraud_ratio)
    
    if is_fraud and fraud_pool:
        tx_raw = fraud_pool[st.session_state.fraud_idx % len(fraud_pool)].copy()
        st.session_state.fraud_idx += 1
    elif legit_pool:
        tx_raw = legit_pool[st.session_state.legit_idx % len(legit_pool)].copy()
        st.session_state.legit_idx += 1
    else:
        # Fallback synthetic record
        tx_raw = {"Time": 50000.0, "Amount": 45.0, "Class": 0}
        for i in range(1, 29):
            tx_raw[f"V{i}"] = 0.0

    tx_raw["transaction_id"] = f"tx_stream_{uuid.uuid4().hex[:8]}"
    return tx_raw

def _process_and_record_transaction(tx_raw: dict, threshold: float) -> dict:
    """Processes a transaction through the model, measures latency, and persists to DB."""
    t_start = time.perf_counter()
    pred_res = predict_single_transaction(tx_raw, threshold=threshold)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Persist to database
    try:
        save_transaction(tx_raw)
        save_prediction(pred_res)
        if pred_res["is_fraud"] or pred_res["risk_level"] in ("HIGH", "CRITICAL"):
            save_alert({
                "transaction_id": pred_res["transaction_id"],
                "amount": pred_res["amount"],
                "fraud_probability": pred_res["fraud_probability"],
                "risk_level": pred_res["risk_level"],
                "alert_status": "NEW"
            })
    except Exception as e:
        print(f"[RealTimeStream] DB persistence notice: {e}")

    record = {
        **pred_res,
        "latency_ms": round(latency_ms, 2),
        "Time": tx_raw["Time"],
        "actual_class": int(tx_raw.get("Class", 0)),
        "timestamp_str": time.strftime("%H:%M:%S")
    }
    return record

def render_real_time_monitoring():
    st.markdown("## 📡 Real-Time Fraud Detection & Alert Stream")
    st.markdown(
        "Live transaction processing console. Every transaction payload is ingested, "
        "preprocessed, scored by the ML model in sub-milliseconds, and evaluated against multi-tiered business risk thresholds."
    )

    # Initialize session history
    if "live_stream_records" not in st.session_state:
        st.session_state.live_stream_records = []

    # 1. Control Panel
    with st.expander("⚙️ Stream Simulation & Ingestion Controls", expanded=True):
        col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
        with col_c1:
            stream_speed = st.slider("Streaming Pace Delay (s)", min_value=0.02, max_value=0.50, value=0.08, step=0.02)
        with col_c2:
            stream_threshold = st.slider("Decision Threshold", min_value=0.10, max_value=0.90, value=0.30, step=0.05,
                                         help="Optimal F1 threshold is 0.30 based on model training.")
        with col_c3:
            fraud_frequency = st.slider("Simulated Attack Rate (%)", min_value=5, max_value=60, value=25, step=5,
                                        help="Percentage of simulated transactions that are fraudulent attacks.")

        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        with btn_c1:
            start_batch = st.button("▶️ Stream 20 Transactions", type="primary", use_container_width=True)
        with btn_c2:
            step_one = st.button("⚡ Stream 1 Transaction", use_container_width=True)
        with btn_c3:
            inject_fraud = st.button("🚨 Inject Fraud Attack Burst (5)", use_container_width=True)
        with btn_c4:
            clear_btn = st.button("🧹 Clear Live Telemetry", use_container_width=True)

    if clear_btn:
        st.session_state.live_stream_records = []
        st.rerun()

    st.markdown("---")

    # Layout Placeholders for Real-Time Updating
    placeholder_kpis = st.empty()
    placeholder_banner = st.empty()
    placeholder_inspector = st.empty()
    placeholder_chart = st.empty()
    placeholder_table = st.empty()

    def update_ui_display(latest_record: dict = None):
        records = st.session_state.live_stream_records
        total = len(records)
        frauds = sum(1 for r in records if r["is_fraud"])
        legit = total - frauds
        fraud_losses_prevented = sum(r["amount"] for r in records if r["is_fraud"])
        avg_latency = (sum(r["latency_ms"] for r in records) / total) if total > 0 else 0.0

        st.session_state.stream_render_tick = st.session_state.get("stream_render_tick", 0) + 1
        render_tick = st.session_state.stream_render_tick

        # 1. Update Session KPIs
        with placeholder_kpis.container():
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Transactions Processed", f"{total:,}", delta="Live Ingestion")
            with k2:
                st.metric("Frauds Intercepted", f"{frauds:,}", delta=f"{frauds} Blocked", delta_color="inverse")
            with k3:
                st.metric("Legitimate Cleared", f"{legit:,}")
            with k4:
                st.metric("Fraud Losses Blocked", f"${fraud_losses_prevented:,.2f}", delta="Protected Capital")
            with k5:
                st.metric("Avg ML Scoring Latency", f"{avg_latency:.2f} ms", delta="Sub-millisecond")

        # 2. Update Live Alert Banner & Inspector Card
        if latest_record:
            risk = latest_record["risk_level"]
            prob = latest_record["fraud_probability"]
            amt = latest_record["amount"]
            tx_id = latest_record["transaction_id"]
            action = latest_record["recommendation"]
            is_fraud = latest_record["is_fraud"]

            if is_fraud or risk in ("HIGH", "CRITICAL"):
                banner_color = "#DC2626"
                banner_bg = "#FEF2F2"
                banner_border = "#EF4444"
                status_icon = "🚨 FRAUD THREAT INTERCEPTED"
                badge_bg = "#EF4444"
            else:
                banner_color = "#166534"
                banner_bg = "#F0FDF4"
                banner_border = "#22C55E"
                status_icon = "✅ TRANSACTION APPROVED"
                badge_bg = "#10B981"

            with placeholder_banner.container():
                st.markdown(
                    f"""
                    <div style="background-color: {banner_bg}; border: 2px solid {banner_border}; padding: 14px 20px; border-radius: 8px; margin-bottom: 15px;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: {banner_color};">
                            {status_icon} | ID: <code>{tx_id}</code> | Amount: <b>${amt:,.2f}</b> | Prob: <b>{prob * 100:.1f}%</b> | Risk: <b>{risk}</b> | Action: <code>{action}</code>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with placeholder_inspector.container():
                st.subheader("Current Transaction Telemetry Inspector")
                col_i1, col_i2, col_i3 = st.columns([2, 3, 2])
                with col_i1:
                    st.markdown(f"**Transaction ID**: `{tx_id}`")
                    st.markdown(f"**Amount**: `${amt:,.2f}`")
                    st.markdown(f"**Arrival Timestamp**: `{latest_record['timestamp_str']}`")
                with col_i2:
                    # Live Probability Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "valueformat": ".1f"},
                        title={"text": "Real-Time Fraud Probability", "font": {"size": 14}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": badge_bg},
                            "steps": [
                                {"range": [0, 30], "color": "#E2E8F0"},
                                {"range": [30, 60], "color": "#FEF3C7"},
                                {"range": [60, 85], "color": "#FFEDD5"},
                                {"range": [85, 100], "color": "#FEE2E2"}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 3},
                                "thickness": 0.8,
                                "value": stream_threshold * 100
                            }
                        }
                    ))
                    fig_g.update_layout(height=170, margin=dict(l=10, r=10, t=25, b=10))
                    st.plotly_chart(fig_g, use_container_width=True, key=f"realtime_gauge_{render_tick}")
                with col_i3:
                    st.markdown(f"**Inference Latency**: `{latest_record['latency_ms']} ms`")
                    st.markdown(f"**Assigned Risk Level**: **{risk}**")
                    st.markdown(f"**Recommended Action**: `{action}`")

        # 3. Update Real-Time Probability Stream Chart
        if records:
            recent_30 = records[-30:]
            fig_stream = go.Figure()
            
            x_indices = list(range(len(records) - len(recent_30) + 1, len(records) + 1))
            probs = [r["fraud_probability"] for r in recent_30]
            colors = ["#DC2626" if r["is_fraud"] else "#2563EB" for r in recent_30]
            amts = [f"${r['amount']:.2f}" for r in recent_30]
            ids = [r["transaction_id"] for r in recent_30]

            fig_stream.add_trace(go.Scatter(
                x=x_indices,
                y=probs,
                mode="lines+markers",
                name="Fraud Probability",
                line=dict(color="#94A3B8", width=1.5),
                marker=dict(size=10, color=colors),
                hovertext=[f"ID: {i}<br>Amt: {a}<br>Prob: {p*100:.1f}%" for i, a, p in zip(ids, amts, probs)],
                hoverinfo="text"
            ))
            fig_stream.add_hline(
                y=stream_threshold,
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=f"Fraud Threshold ({stream_threshold})",
                annotation_position="bottom right"
            )
            fig_stream.update_layout(
                title="Live Fraud Probability Stream Timeline (Last 30 Events)",
                xaxis_title="Stream Event Sequence #",
                yaxis_title="Estimated Probability",
                yaxis=dict(range=[-0.05, 1.05]),
                height=260,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            placeholder_chart.plotly_chart(fig_stream, use_container_width=True, key=f"realtime_stream_chart_{render_tick}")

        # 4. Update Live Stream Table
        if records:
            df_stream = pd.DataFrame(records[-12:][::-1])[
                ["transaction_id", "timestamp_str", "amount", "fraud_probability", "risk_level", "is_fraud", "recommendation", "latency_ms"]
            ].copy()
            df_stream.columns = ["Tx ID", "Time", "Amount ($)", "Fraud Prob", "Risk Level", "Verdict", "Action", "Latency (ms)"]
            df_stream["Amount ($)"] = df_stream["Amount ($)"].map("${:,.2f}".format)
            df_stream["Fraud Prob"] = df_stream["Fraud Prob"].map("{:.1%}".format)
            df_stream["Verdict"] = df_stream["Verdict"].map(lambda f: "🚨 FRAUD" if f else "✅ LEGIT")
            
            with placeholder_table.container():
                st.markdown("**Live Ingested Transactions (Most Recent First)**")
                st.dataframe(df_stream, use_container_width=True, hide_index=True)

    # Initial render of existing session records (if any)
    if st.session_state.live_stream_records:
        update_ui_display(st.session_state.live_stream_records[-1])
    else:
        with placeholder_banner.container():
            st.info("💡 **Ready to Stream**: Click **'▶️ Stream 20 Transactions'** or **'⚡ Stream 1 Transaction'** to watch live transactions, ML probability scoring, and fraud detections animate across the screen.")

    # 2. Execution Triggers
    if start_batch:
        n_tx = 20
        for i in range(n_tx):
            tx_data = _fetch_next_transaction(force_fraud=False, fraud_ratio=(fraud_frequency / 100.0))
            record = _process_and_record_transaction(tx_data, threshold=stream_threshold)
            st.session_state.live_stream_records.append(record)
            update_ui_display(record)
            time.sleep(stream_speed)
        st.toast(f"Completed streaming batch of {n_tx} transactions!", icon="🚀")

    elif step_one:
        tx_data = _fetch_next_transaction(force_fraud=False, fraud_ratio=(fraud_frequency / 100.0))
        record = _process_and_record_transaction(tx_data, threshold=stream_threshold)
        st.session_state.live_stream_records.append(record)
        update_ui_display(record)
        st.toast(f"Processed single transaction: {record['transaction_id']}", icon="⚡")

    elif inject_fraud:
        for i in range(5):
            tx_data = _fetch_next_transaction(force_fraud=True)
            record = _process_and_record_transaction(tx_data, threshold=stream_threshold)
            st.session_state.live_stream_records.append(record)
            update_ui_display(record)
            time.sleep(stream_speed)
        st.toast("Injected 5 consecutive high-risk fraud attacks!", icon="🚨")

    # 3. Database Historical Tab Navigation
    st.markdown("---")
    st.subheader("Persistent Relational Database Audit Records")
    tab_alerts, tab_db_tx = st.tabs(["🚨 Persisted Operational Alerts", "📋 Relational Transaction Log"])

    with tab_alerts:
        db_alerts = get_recent_alerts(limit=50)
        if not db_alerts.empty:
            st.dataframe(
                db_alerts.style.format({
                    "amount": "${:,.2f}",
                    "fraud_probability": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No persistent alerts found in database yet.")

    with tab_db_tx:
        db_tx = get_recent_transactions(limit=50)
        if not db_tx.empty:
            st.dataframe(
                db_tx.style.format({
                    "amount": "${:,.2f}",
                    "fraud_probability": lambda p: f"{p:.2%}" if pd.notnull(p) else "N/A"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No transactions found in database.")
