"""
Dashboard Page 6: SQL Business Analytics
Executes advanced SQL queries (CTEs, Window Functions, Aggregations)
against the relational database and provides an interactive query console.
"""

import time
import streamlit as st
import plotly.express as px
import pandas as pd
from src.database.queries import run_raw_sql

PRESET_SQL_QUERIES = {
    "1. Overall Fraud Volume & Rate": """SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_percentage,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS fraud_dollar_loss
FROM transactions;""",

    "2. Average Ticket Size (Legitimate vs Fraudulent)": """SELECT 
    CASE WHEN class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_type,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount,
    ROUND(SUM(amount), 2) AS total_volume
FROM transactions
GROUP BY class;""",

    "3. Cohort Analysis: Fraud Rate by Amount Bracket (CTE + CASE)": """WITH amount_brackets AS (
    SELECT 
        amount,
        class,
        CASE 
            WHEN amount < 20.0 THEN '1. Micro (< $20)'
            WHEN amount >= 20.0 AND amount < 100.0 THEN '2. Small ($20 - $100)'
            WHEN amount >= 100.0 AND amount < 500.0 THEN '3. Medium ($100 - $500)'
            WHEN amount >= 500.0 AND amount < 2000.0 THEN '4. Large ($500 - $2,000)'
            ELSE '5. Jumbo ($2,000+)'
        END AS amount_tier
    FROM transactions
)
SELECT 
    amount_tier,
    COUNT(*) AS total_tx,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_tx,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_volume,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS fraud_loss
FROM amount_brackets
GROUP BY amount_tier
ORDER BY amount_tier;""",

    "4. Hourly Fraud Velocity & Risk (Floor Division CTE)": """WITH hourly_tx AS (
    SELECT 
        CAST((time / 3600) AS INTEGER) % 24 AS hour_of_day,
        class,
        amount
    FROM transactions
)
SELECT 
    hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS hourly_volume
FROM hourly_tx
GROUP BY hour_of_day
ORDER BY hour_of_day;""",

    "5. Top 15 Highest-Value Fraudulent Transactions": """SELECT 
    transaction_id,
    ROUND(time / 3600.0, 2) AS elapsed_hours,
    amount,
    class
FROM transactions
WHERE class = 1
ORDER BY amount DESC
LIMIT 15;""",

    "6. Suspicious Periods & Density Ranking (Window Functions)": """WITH period_summary AS (
    SELECT 
        CAST((time / 7200) AS INTEGER) AS two_hour_block,
        COUNT(*) AS tx_count,
        SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_count,
        ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS fraud_loss
    FROM transactions
    GROUP BY CAST((time / 7200) AS INTEGER)
)
SELECT 
    two_hour_block * 2 AS block_start_hour,
    (two_hour_block * 2) + 2 AS block_end_hour,
    tx_count,
    fraud_count,
    fraud_loss,
    ROUND(100.0 * fraud_count / tx_count, 4) AS fraud_rate_pct,
    RANK() OVER (ORDER BY fraud_count DESC) AS fraud_density_rank,
    SUM(fraud_count) OVER (ORDER BY two_hour_block) AS cumulative_fraud
FROM period_summary
ORDER BY fraud_density_rank ASC
LIMIT 10;""",

    "7. Operational Alert Risk Tiers Summary": """SELECT 
    risk_level,
    COUNT(*) AS alert_count,
    ROUND(AVG(amount), 2) AS avg_alert_amount,
    ROUND(SUM(amount), 2) AS total_alert_exposure,
    ROUND(AVG(fraud_probability) * 100, 2) AS avg_confidence_pct
FROM fraud_alerts
GROUP BY risk_level
ORDER BY 
    CASE risk_level
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'MEDIUM' THEN 3
        WHEN 'LOW' THEN 4
        ELSE 5
    END;"""
}

def render_sql_analytics():
    st.markdown("## 🗄️ SQL Business Analytics & KPI Engine")
    st.markdown(
        "Demonstrating relational database analytics, CTEs, Window Functions (`RANK()`, cumulative sums), "
        "and cohort aggregations executed directly against the live SQL database."
    )

    tab_presets, tab_custom = st.tabs(["📊 Pre-Built Business KPI Queries", "💻 Live Custom SQL Console"])

    with tab_presets:
        query_name = st.selectbox("Select Business Analysis Query", options=list(PRESET_SQL_QUERIES.keys()))
        sql_code = PRESET_SQL_QUERIES[query_name]

        st.code(sql_code, language="sql")

        col_exec, col_info = st.columns([1, 4])
        with col_exec:
            execute_btn = st.button("🚀 Execute Query", key="exec_preset", type="primary")

        if execute_btn or True:  # Run by default
            start_t = time.time()
            try:
                result_df = run_raw_sql(sql_code)
                elapsed_ms = (time.time() - start_t) * 1000

                st.success(f"Executed in {elapsed_ms:.1f} ms. Returned {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Quick Chart for multi-row numeric queries
                if len(result_df) > 1 and ("hour_of_day" in result_df.columns or "amount_tier" in result_df.columns or "block_start_hour" in result_df.columns):
                    x_col = "hour_of_day" if "hour_of_day" in result_df.columns else ("amount_tier" if "amount_tier" in result_df.columns else "block_start_hour")
                    y_col = "fraud_rate_pct" if "fraud_rate_pct" in result_df.columns else ("fraud_count" if "fraud_count" in result_df.columns else result_df.columns[1])
                    
                    fig = px.bar(result_df, x=x_col, y=y_col, title=f"{query_name} ({y_col} vs {x_col})", color=y_col, color_continuous_scale="Blues")
                    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"SQL execution error: {e}")

    with tab_custom:
        st.markdown("### Execute Custom Read Queries Against Relational Database")
        custom_sql = st.text_area(
            "Enter SQL Query (Read-Only `SELECT`)",
            value="SELECT transaction_id, time, amount, class FROM transactions WHERE class = 1 LIMIT 10;",
            height=120
        )
        if st.button("⚡ Run Custom Query", key="exec_custom", type="primary"):
            cleaned_query = custom_sql.strip().lower()
            if not cleaned_query.startswith("select") and not cleaned_query.startswith("with"):
                st.error("Only SELECT and WITH analytical queries are permitted in the console.")
            else:
                start_t = time.time()
                try:
                    res_df = run_raw_sql(custom_sql)
                    elapsed_ms = (time.time() - start_t) * 1000
                    st.success(f"Executed in {elapsed_ms:.1f} ms. Returned {len(res_df)} rows.")
                    st.dataframe(res_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Execution failed: {e}")
