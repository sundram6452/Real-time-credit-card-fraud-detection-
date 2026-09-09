"""
Business Fraud Analytics Module
Calculates aggregate KPIs, cohort analyses, and temporal patterns
for executive reporting and dashboard visualization.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def calculate_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level executive metrics from transaction dataset.
    """
    total_tx = len(df)
    if total_tx == 0:
        return {
            "total_transactions": 0,
            "fraud_transactions": 0,
            "fraud_rate": 0.0,
            "total_amount": 0.0,
            "fraud_amount": 0.0,
            "avg_amount": 0.0,
            "avg_fraud_amount": 0.0,
            "legit_amount": 0.0
        }

    is_fraud = (df["Class"] == 1) if "Class" in df.columns else pd.Series([False] * total_tx)
    fraud_tx = int(is_fraud.sum())
    fraud_rate = (fraud_tx / total_tx * 100.0)
    
    total_amt = float(df["Amount"].sum())
    fraud_amt = float(df[is_fraud]["Amount"].sum()) if fraud_tx > 0 else 0.0
    legit_amt = total_amt - fraud_amt
    avg_amt = float(df["Amount"].mean())
    avg_fraud_amt = float(df[is_fraud]["Amount"].mean()) if fraud_tx > 0 else 0.0

    return {
        "total_transactions": total_tx,
        "fraud_transactions": fraud_tx,
        "fraud_rate": round(fraud_rate, 4),
        "total_amount": round(total_amt, 2),
        "fraud_amount": round(fraud_amt, 2),
        "avg_amount": round(avg_amt, 2),
        "avg_fraud_amount": round(avg_fraud_amt, 2),
        "legit_amount": round(legit_amt, 2)
    }

def get_hourly_fraud_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates transaction and fraud volume by hour of day (0-23)."""
    df_calc = df.copy()
    df_calc["hour"] = ((df_calc["Time"] // 3600) % 24).astype(int)
    
    grouped = df_calc.groupby("hour").agg(
        total_transactions=("Amount", "count"),
        fraud_transactions=("Class", "sum"),
        total_amount=("Amount", "sum"),
        fraud_amount=("Amount", lambda s: df_calc.loc[s.index[df_calc.loc[s.index, "Class"] == 1], "Amount"].sum())
    ).reset_index()

    grouped["fraud_rate_pct"] = (grouped["fraud_transactions"] / grouped["total_transactions"] * 100.0).round(4)
    return grouped

def get_amount_bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Segments transactions into standard dollar brackets to identify fraud concentration."""
    bins = [-1.0, 20.0, 100.0, 500.0, 2000.0, float("inf")]
    labels = ["< $20", "$20 - $100", "$100 - $500", "$500 - $2,000", "$2,000+"]
    
    df_calc = df.copy()
    df_calc["tier"] = pd.cut(df_calc["Amount"], bins=bins, labels=labels)
    
    grouped = df_calc.groupby("tier", observed=False).agg(
        total_transactions=("Amount", "count"),
        fraud_transactions=("Class", "sum"),
        total_amount=("Amount", "sum"),
        fraud_amount=("Amount", lambda s: df_calc.loc[s.index[df_calc.loc[s.index, "Class"] == 1], "Amount"].sum())
    ).reset_index()

    grouped["fraud_rate_pct"] = (
        grouped["fraud_transactions"] / grouped["total_transactions"] * 100.0
    ).fillna(0.0).round(4)
    
    return grouped

def get_top_fraud_transactions(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Returns highest value fraudulent transactions."""
    if "Class" not in df.columns:
        return pd.DataFrame()
    fraud_df = df[df["Class"] == 1].sort_values(by="Amount", ascending=False).head(top_n).copy()
    fraud_df["hours_elapsed"] = (fraud_df["Time"] / 3600.0).round(2)
    display_cols = ["hours_elapsed", "Amount", "Class"] + [f"V{i}" for i in [14, 10, 12, 17, 4]]
    valid_cols = [c for c in display_cols if c in fraud_df.columns]
    return fraud_df[valid_cols]
