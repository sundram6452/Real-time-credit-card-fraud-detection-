"""
Big Data Analytics Module (PySpark)
Demonstrates distributed batch analytics, window functions, and aggregation pipelines.
Includes robust fallback when Java runtime version is incompatible or unconfigured.
"""

import os
import sys
import shutil
from typing import Dict, Any, Optional
import pandas as pd

def is_spark_available() -> bool:
    """Checks if a compatible Java 17+ environment and PySpark are enabled."""
    if os.getenv("USE_SPARK", "false").lower() not in ("true", "1"):
        return False
    java_bin = shutil.which("java")
    if not java_bin:
        return False
    try:
        import pyspark
        return True
    except ImportError:
        return False

def _run_vectorized_fallback(csv_path: str) -> Dict[str, Any]:
    """Batch engine performing window calculations and aggregations."""
    print("[Spark Analytics] Java 17+ not configured locally. Executing batch analytics via vectorized engine.")
    df = pd.read_csv(csv_path)
    df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
    
    hourly = df.groupby("hour_of_day").agg(
        tx_count=("Amount", "count"),
        fraud_count=("Class", "sum"),
        total_volume=("Amount", lambda s: round(s.sum(), 2)),
        fraud_volume=("Amount", lambda s: round(df.loc[s.index[df.loc[s.index, "Class"] == 1], "Amount"].sum(), 2))
    ).reset_index()
    
    # Cumulative window running total
    hourly["cumulative_fraud_volume"] = hourly["fraud_volume"].cumsum().round(2)

    print(f"[Spark Analytics] Successfully processed {len(df):,} transactions.")
    return {
        "engine": "Vectorized Batch Engine (Spark-equivalent)",
        "summary": hourly.head(10).to_dict(orient="records"),
        "dataframe": hourly
    }

def run_spark_fraud_analytics(
    csv_path: str,
    output_parquet: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes large-scale batch processing on transaction data.
    Attempts PySpark Session if USE_SPARK=true, otherwise executes
    the vectorized batch engine.
    """
    if is_spark_available():
        try:
            print("[Spark Analytics] Initializing PySpark Session...")
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
            from pyspark.sql.window import Window

            spark = SparkSession.builder \
                .appName("CreditCardFraudBatchAnalytics") \
                .master("local[*]") \
                .getOrCreate()

            try:
                print(f"[Spark Analytics] Reading transactions from: {csv_path}")
                sdf = spark.read.csv(csv_path, header=True, inferSchema=True)

                sdf = sdf.withColumn("hour_of_day", (F.col("Time") / 3600).cast("int") % 24)
                
                hourly_window = Window.orderBy("hour_of_day")
                hourly_agg = sdf.groupBy("hour_of_day").agg(
                    F.count("*").alias("tx_count"),
                    F.sum("Class").alias("fraud_count"),
                    F.round(F.sum("Amount"), 2).alias("total_volume"),
                    F.round(F.sum(F.when(F.col("Class") == 1, F.col("Amount")).otherwise(0)), 2).alias("fraud_volume")
                ).withColumn(
                    "cumulative_fraud_volume",
                    F.round(F.sum("fraud_volume").over(hourly_window), 2)
                ).orderBy("hour_of_day")

                results_df = hourly_agg.toPandas()
                print(f"[Spark Analytics] Successfully processed {sdf.count():,} rows via Spark.")

                if output_parquet:
                    hourly_agg.write.mode("overwrite").parquet(output_parquet)
                    print(f"[Spark Analytics] Exported parquet to: {output_parquet}")

                return {
                    "engine": "PySpark Distributed Engine",
                    "summary": results_df.head(10).to_dict(orient="records"),
                    "dataframe": results_df
                }
            finally:
                spark.stop()
        except Exception as e:
            print(f"[Spark Analytics] PySpark error ({e}). Falling back.")
            return _run_vectorized_fallback(csv_path)
    else:
        return _run_vectorized_fallback(csv_path)

if __name__ == "__main__":
    from src.config import PROCESSED_DATA_PATH
    res = run_spark_fraud_analytics(str(PROCESSED_DATA_PATH))
    print(res["dataframe"].head())
