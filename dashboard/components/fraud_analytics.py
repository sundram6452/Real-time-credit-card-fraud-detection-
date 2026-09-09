"""
Dashboard Page 2: Fraud Analytics
In-depth analysis of fraud occurrence by hour, amount buckets, and high-value incidents.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.analytics.fraud_analytics import (
    get_hourly_fraud_patterns,
    get_amount_bucket_analysis,
    get_top_fraud_transactions
)

def render_fraud_analytics(df: pd.DataFrame):
    st.markdown("## 🔍 Deep-Dive Fraud Analytics")
    st.markdown("Granular behavioral patterns, transaction size cohorts, and temporal risk concentrations.")

    # 1. Hourly Fraud Distribution
    st.subheader("Hourly Fraud Patterns & Peak Risk Hours")
    hourly_df = get_hourly_fraud_patterns(df)

    fig_hour = go.Figure()
    fig_hour.add_trace(go.Bar(
        x=hourly_df["hour"],
        y=hourly_df["total_transactions"],
        name="Total Transaction Volume",
        marker_color="#94A3B8",
        opacity=0.6,
        yaxis="y"
    ))
    fig_hour.add_trace(go.Scatter(
        x=hourly_df["hour"],
        y=hourly_df["fraud_rate_pct"],
        name="Fraud Rate (%)",
        mode="lines+markers",
        line=dict(color="#DC2626", width=3),
        marker=dict(size=8),
        yaxis="y2"
    ))
    fig_hour.update_layout(
        xaxis=dict(title="Hour of Day (0 - 23)", tickmode="linear", dtick=1),
        yaxis=dict(title="Total Transactions", side="left", showgrid=False),
        yaxis2=dict(title="Fraud Rate (%)", side="right", overlaying="y", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=380
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # Key takeaway alert
    peak_hour_row = hourly_df.loc[hourly_df["fraud_rate_pct"].idxmax()]
    st.info(
        f"**Business Insight**: Peak fraud incidence occurs around **Hour {int(peak_hour_row['hour']):02d}:00** "
        f"with a fraud rate of **{peak_hour_row['fraud_rate_pct']:.2f}%** ({int(peak_hour_row['fraud_transactions'])} fraud events), "
        "correlating with low legitimate transaction traffic and early morning hours."
    )

    st.markdown("---")

    # 2. Amount Bracket Analysis
    st.subheader("Fraud Rate by Transaction Size Cohort")
    cohort_df = get_amount_bucket_analysis(df)
    
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        fig_cohort = px.bar(
            cohort_df,
            x="tier",
            y="fraud_rate_pct",
            text="fraud_rate_pct",
            color="fraud_rate_pct",
            color_continuous_scale="Reds",
            labels={"tier": "Ticket Size Tier", "fraud_rate_pct": "Fraud Rate (%)"},
            title="Fraud Probability across Ticket Size Tiers"
        )
        fig_cohort.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_cohort.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_cohort, use_container_width=True)

    with col_c2:
        st.markdown("**Cohort Financial Exposure Breakdown**")
        cohort_display = cohort_df[["tier", "total_transactions", "fraud_transactions", "fraud_amount"]].copy()
        cohort_display.columns = ["Tier", "Total Tx", "Fraud Count", "Fraud Loss ($)"]
        cohort_display["Fraud Loss ($)"] = cohort_display["Fraud Loss ($)"].map("${:,.2f}".format)
        cohort_display["Total Tx"] = cohort_display["Total Tx"].map("{:,}".format)
        st.dataframe(cohort_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Top High-Value Fraud Transactions
    st.subheader("🚨 Top 15 Highest-Value Confirmed Fraud Incidents")
    top_frauds = get_top_fraud_transactions(df, top_n=15)
    if not top_frauds.empty:
        st.dataframe(
            top_frauds.style.format({
                "Amount": "${:,.2f}",
                "hours_elapsed": "{:.1f} hrs",
                "V14": "{:.2f}",
                "V10": "{:.2f}",
                "V12": "{:.2f}",
                "V17": "{:.2f}",
                "V4": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )
