"""
Big data batch processing module.
"""
from src.bigdata.spark_analytics import run_spark_fraud_analytics, is_spark_available

__all__ = ["run_spark_fraud_analytics", "is_spark_available"]
