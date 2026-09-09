"""
Analytics package initialization.
"""
from src.analytics.fraud_analytics import (
    calculate_executive_kpis,
    get_hourly_fraud_patterns,
    get_amount_bucket_analysis,
    get_top_fraud_transactions
)

__all__ = [
    "calculate_executive_kpis",
    "get_hourly_fraud_patterns",
    "get_amount_bucket_analysis",
    "get_top_fraud_transactions"
]
