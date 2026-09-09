"""
FraudGuard AI - Real-Time Credit Card Fraud Detection & Analytics Platform
Main Streamlit Application
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.config import USE_KAFKA, DATABASE_URL
from src.data.loader import load_transaction_data
from dashboard.components import (
    render_executive_overview,
    render_fraud_analytics,
    render_real_time_monitoring,
    render_model_performance,
    render_transaction_prediction,
    render_sql_analytics
)

# 1. Page Configuration
st.set_page_config(
    page_title="FraudGuard AI | Enterprise Fraud Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Enterprise Look & Feel
st.markdown("""
<style>
    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #475569;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }
    /* Header Brand Styling */
    .brand-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E3A8A;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0px;
    }
    .brand-sub {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: -5px;
        margin-bottom: 20px;
    }
    .badge-status {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 6px;
    }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
</style>
""", unsafe_allow_html=True)

# 3. Data Ingestion Cache
@st.cache_data(show_spinner="Loading transaction repository...")
def get_cached_transactions():
    return load_transaction_data(source="auto", clean=True)

# 4. Sidebar Navigation & Global Controls
with st.sidebar:
    st.markdown("""
    <div class="brand-title">🛡️ FraudGuard AI</div>
    <div class="brand-sub">Real-Time Detection & Analytics Engine</div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "Select Portal View:",
        options=[
            "1. Executive Overview",
            "2. Fraud Analytics",
            "3. Real-Time Monitoring",
            "4. Model Performance",
            "5. Transaction Prediction",
            "6. SQL Analytics"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### System Architecture")
    db_mode = "PostgreSQL" if "postgres" in DATABASE_URL.lower() else "SQLite (Local Embedded)"
    stream_mode = "Apache Kafka" if USE_KAFKA else "Kafka-Compatible Simulator"
    
    st.markdown(f"**Database**: <span class='badge-status badge-blue'>{db_mode}</span>", unsafe_allow_html=True)
    st.markdown(f"**Streaming**: <span class='badge-status badge-green'>{stream_mode}</span>", unsafe_allow_html=True)
    st.markdown("**Active Model**: `RandomForestClassifier (v1.0)`")
    st.markdown("**Decision Threshold**: `0.30 (Tuned F1)`")

    st.markdown("---")
    st.markdown("### Dataset Filters")
    min_amt, max_amt = st.slider(
        "Filter Amount Range ($)",
        min_value=0.0,
        max_value=5000.0,
        value=(0.0, 3000.0),
        step=50.0
    )

# 5. Load Data & Apply Global Filters
df_all = get_cached_transactions()
df_filtered = df_all[(df_all["Amount"] >= min_amt) & (df_all["Amount"] <= max_amt)].copy()

# 6. Page Routing
if page == "1. Executive Overview":
    render_executive_overview(df_filtered)
elif page == "2. Fraud Analytics":
    render_fraud_analytics(df_filtered)
elif page == "3. Real-Time Monitoring":
    render_real_time_monitoring()
elif page == "4. Model Performance":
    render_model_performance()
elif page == "5. Transaction Prediction":
    render_transaction_prediction()
elif page == "6. SQL Analytics":
    render_sql_analytics()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "FraudGuard AI Platform | Production-Grade Credit Card Fraud Detection, Streaming & Analytics System"
    "</div>",
    unsafe_allow_html=True
)
