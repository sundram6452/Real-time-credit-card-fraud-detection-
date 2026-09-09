"""
Dashboard Page 3: Real-Time Fraud Monitoring
Live incoming transaction stream, dynamic risk categorization, and operational fraud alerts.
"""

import time
import streamlit as st
import pandas as pd
from src.config import DEFAULT_FRAUD_THRESHOLD
from src.database.queries import get_recent_alerts, get_recent_transactions
from src.streaming.producer import start_producing_transactions
from src.streaming.consumer import start_consuming_transactions

def render_real_time_monitoring():
    st.markdown("## 📡 Real-Time Fraud Detection & Alert Stream")
    st.markdown(
        "Event-driven transaction ingestion pipeline. Each transaction is validated, "
        "preprocessed, scored by the ML model in sub-milliseconds, and evaluated against multi-tiered business risk thresholds."
    )

    # Stream Controls Bar
    with st.expander("⚙️ Stream Ingestion Controls & Simulation", expanded=True):
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 2, 2, 2])
        
        with col_ctrl1:
            batch_size = st.selectbox("Stream Batch Size", options=[10, 25, 50, 100], index=1)
        with col_ctrl2:
            delay = st.slider("Inter-event Delay (s)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
        with col_ctrl3:
            fraud_ratio = st.slider("Simulated Fraud Ratio", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
        with col_ctrl4:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶️ Stream Transactions Now", type="primary", use_container_width=True)

    # Ingestion Trigger
    if run_btn:
        import threading
        progress_bar = st.progress(0, text="Initializing Kafka-compatible stream pipeline...")
        
        # Start consumer in thread
        consumer_events = []
        def on_event(payload):
            consumer_events.append(payload)

        cons_thread = threading.Thread(
            target=start_consuming_transactions,
            kwargs={
                "max_records": batch_size,
                "poll_timeout_ms": 500,
                "callback": on_event
            },
            daemon=True
        )
        cons_thread.start()
        time.sleep(0.2)

        # Produce events
        start_producing_transactions(
            max_records=batch_size,
            delay=delay,
            interleave_fraud_ratio=fraud_ratio
        )
        cons_thread.join(timeout=10)

        progress_bar.progress(100, text=f"Successfully streamed and scored {len(consumer_events)} events!")
        st.success(f"Streamed {len(consumer_events)} transactions. Database and alert tables updated!")

    # Live Status & Metrics from DB
    st.markdown("---")
    alerts_df = get_recent_alerts(limit=50)
    recent_tx_df = get_recent_transactions(limit=50)

    total_ingested = len(recent_tx_df)
    total_alerts = len(alerts_df)
    critical_alerts = len(alerts_df[alerts_df["risk_level"] == "CRITICAL"]) if not alerts_df.empty else 0
    high_alerts = len(alerts_df[alerts_df["risk_level"] == "HIGH"]) if not alerts_df.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Live Stream Ingested", f"{total_ingested} Tx")
    with m2:
        st.metric("Triggered Fraud Alerts", f"{total_alerts}")
    with m3:
        st.metric("Critical Risk Alerts", f"{critical_alerts}", delta=f"{critical_alerts} Urgent", delta_color="inverse")
    with m4:
        st.metric("High Risk Alerts", f"{high_alerts}")

    # Alert Feed and Transaction Feed Tabs
    tab_alerts, tab_feed = st.tabs(["🚨 Operational Fraud Alerts Feed", "📋 Live Transaction Ingestion Log"])

    with tab_alerts:
        if not alerts_df.empty:
            # Highlight high and critical risk
            def highlight_risk(val):
                if val == "CRITICAL":
                    return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
                elif val == "HIGH":
                    return "background-color: #FFEDD5; color: #9A3412; font-weight: bold;"
                elif val == "MEDIUM":
                    return "background-color: #FEF3C7; color: #92400E;"
                return "background-color: #DCFCE7; color: #166534;"

            apply_fn = getattr(alerts_df.style, "map", None) or getattr(alerts_df.style, "applymap")
            styled_alerts = apply_fn(highlight_risk, subset=["risk_level"]).format({
                "amount": "${:,.2f}",
                "fraud_probability": "{:.2%}"
            })
            st.dataframe(styled_alerts, use_container_width=True, hide_index=True)
        else:
            st.info("No active fraud alerts recorded yet. Click 'Stream Transactions Now' above to trigger live streaming events.")

    with tab_feed:
        if not recent_tx_df.empty:
            st.dataframe(
                recent_tx_df.style.format({
                    "amount": "${:,.2f}",
                    "fraud_probability": lambda p: f"{p:.2%}" if pd.notnull(p) else "N/A"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No transactions logged in database yet.")
