"""
Dashboard Page 1: Executive Overview
High-level KPIs, macro transaction trends, fraud proportions, and amount distributions.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.analytics.fraud_analytics import calculate_executive_kpis

def render_executive_overview(df: pd.DataFrame):
    st.markdown("## 📊 Executive Fraud Overview")
    st.markdown("Macro portfolio view of transaction volume, fraud rate, and direct financial exposure.")

    kpis = calculate_executive_kpis(df)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{kpis['total_transactions']:,}",
            help="Total number of evaluated transaction records."
        )
    with col2:
        st.metric(
            label="Fraudulent Transactions",
            value=f"{kpis['fraud_transactions']:,}",
            delta=f"{kpis['fraud_rate']:.3f}% Rate",
            delta_color="inverse",
            help="Count of verified fraudulent transactions."
        )
    with col3:
        st.metric(
            label="Total Volume Handled",
            value=f"${kpis['total_amount']:,.2f}",
            help="Cumulative gross dollar volume processed."
        )
    with col4:
        st.metric(
            label="Identified Fraud Exposure",
            value=f"${kpis['fraud_amount']:,.2f}",
            delta=f"${kpis['avg_fraud_amount']:.2f} Avg",
            delta_color="inverse",
            help="Total dollar value associated with fraudulent transactions."
        )

    st.markdown("---")

    # Visual Charts
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.subheader("Transaction Volume & Fraud Over Time")
        # Bin transactions into 2-hour blocks
        df_time = df.copy()
        df_time["two_hour_block"] = (df_time["Time"] // 7200).astype(int)
        time_agg = df_time.groupby(["two_hour_block", "Class"]).size().unstack(fill_value=0).reset_index()
        time_agg.columns = ["block", "Legitimate", "Fraud"]
        time_agg["Hour_Elapsed"] = time_agg["block"] * 2

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=time_agg["Hour_Elapsed"],
            y=time_agg["Legitimate"],
            name="Legitimate Transactions",
            marker_color="#1E3A8A",
            opacity=0.85
        ))
        fig_trend.add_trace(go.Scatter(
            x=time_agg["Hour_Elapsed"],
            y=time_agg["Fraud"] * 100,  # Scaled for visibility
            name="Fraud Volume (x100 for visibility)",
            mode="lines+markers",
            line=dict(color="#EF4444", width=3)
        ))
        fig_trend.update_layout(
            barmode="overlay",
            xaxis_title="Simulation Hours Elapsed",
            yaxis_title="Transaction Volume",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=360
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.subheader("Class Distribution Proportion")
        fig_donut = px.pie(
            values=[kpis["total_transactions"] - kpis["fraud_transactions"], kpis["fraud_transactions"]],
            names=["Legitimate (0)", "Fraud (1)"],
            hole=0.65,
            color=["Legitimate (0)", "Fraud (1)"],
            color_discrete_map={"Legitimate (0)": "#1E3A8A", "Fraud (1)": "#EF4444"}
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
            showlegend=False,
            annotations=[dict(text=f"{kpis['fraud_rate']:.3f}%<br>Fraud", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Bottom Row: Amount Distributions
    st.subheader("Transaction Amount Profiles by Class")
    col_amt1, col_amt2 = st.columns(2)
    
    with col_amt1:
        fig_box = px.box(
            df[df["Amount"] <= 1000],
            x="Class",
            y="Amount",
            color="Class",
            labels={"Class": "Transaction Type (0=Legit, 1=Fraud)"},
            color_discrete_map={0: "#1E3A8A", 1: "#EF4444"},
            title="Ticket Size Boxplot (Amounts <= $1,000)"
        )
        fig_box.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_amt2:
        # Comparison Metrics Table
        legit_series = df[df["Class"] == 0]["Amount"]
        fraud_series = df[df["Class"] == 1]["Amount"]
        summary_table = pd.DataFrame({
            "Metric": ["Average Ticket", "Median Ticket", "95th Percentile", "Maximum Ticket", "Total Loss / Exposure"],
            "Legitimate (0)": [
                f"${legit_series.mean():.2f}",
                f"${legit_series.median():.2f}",
                f"${legit_series.quantile(0.95):.2f}",
                f"${legit_series.max():.2f}",
                f"${legit_series.sum():,.2f}"
            ],
            "Fraudulent (1)": [
                f"${fraud_series.mean():.2f}",
                f"${fraud_series.median():.2f}",
                f"${fraud_series.quantile(0.95):.2f}",
                f"${fraud_series.max():.2f}",
                f"${fraud_series.sum():,.2f}"
            ]
        })
        st.markdown("**Amount Summary Comparison**")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
