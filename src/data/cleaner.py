"""
Data Cleaner Pipeline
Handles deduplication, missing values inspection, outlier analysis,
and prints standardized cleaning audit metrics.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.validator import validate_dataframe

def clean_transaction_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    output_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and audits credit card transaction data.
    Preserves legitimate and fraudulent outliers in Amount without blind clipping.
    """
    original_rows = len(df)
    
    # 1. Validation check
    val_report = validate_dataframe(df, require_class="Class" in df.columns)
    if not val_report["is_valid"]:
        print("Warning: Validation warnings encountered before cleaning:")
        for err in val_report["errors"]:
            print(f"  - {err}")

    # 2. Missing values check & handling
    missing_values = int(df.isnull().sum().sum())
    if missing_values > 0:
        # Drop rows with missing values if any
        df = df.dropna().copy()
    else:
        df = df.copy()

    # 3. Duplicate detection & removal
    duplicate_rows = int(df.duplicated().sum())
    if remove_duplicates and duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    
    cleaned_rows = len(df)

    # 4. Class distribution analysis
    if "Class" in df.columns:
        fraud_tx = int((df["Class"] == 1).sum())
        legit_tx = int((df["Class"] == 0).sum())
        fraud_pct = (fraud_tx / cleaned_rows * 100.0) if cleaned_rows > 0 else 0.0
    else:
        fraud_tx = 0
        legit_tx = cleaned_rows
        fraud_pct = 0.0

    # 5. Outlier analysis on Amount (without blind deletion)
    amount_stats = {
        "mean": float(df["Amount"].mean()),
        "std": float(df["Amount"].std()),
        "median": float(df["Amount"].median()),
        "p95": float(df["Amount"].quantile(0.95)),
        "p99": float(df["Amount"].quantile(0.99)),
        "max": float(df["Amount"].max()),
    }
    if "Class" in df.columns and fraud_tx > 0:
        amount_stats["fraud_mean"] = float(df[df["Class"] == 1]["Amount"].mean())
        amount_stats["fraud_max"] = float(df[df["Class"] == 1]["Amount"].max())
        amount_stats["legit_mean"] = float(df[df["Class"] == 0]["Amount"].mean())
        amount_stats["legit_max"] = float(df[df["Class"] == 0]["Amount"].max())

    stats: Dict[str, Any] = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "fraud_transactions": fraud_tx,
        "legitimate_transactions": legit_tx,
        "fraud_percentage": fraud_pct,
        "amount_stats": amount_stats
    }

    # Print standardized audit log
    print("=" * 45)
    print("        DATA CLEANING AUDIT REPORT")
    print("=" * 45)
    print(f"Original Rows:           {stats['original_rows']:,}")
    print(f"Cleaned Rows:            {stats['cleaned_rows']:,}")
    print(f"Duplicate Rows:          {stats['duplicate_rows']:,}")
    print(f"Missing Values:          {stats['missing_values']:,}")
    print(f"Fraud Transactions:      {stats['fraud_transactions']:,}")
    print(f"Legitimate Transactions: {stats['legitimate_transactions']:,}")
    print(f"Fraud Percentage:        {stats['fraud_percentage']:.4f}%")
    print("=" * 45)
    print(f"Amount 99th Percentile:  ${amount_stats['p99']:.2f} | Max: ${amount_stats['max']:.2f}")
    if "fraud_mean" in amount_stats:
        print(f"Fraud Mean Amount:       ${amount_stats['fraud_mean']:.2f} (Max: ${amount_stats['fraud_max']:.2f})")
        print(f"Legit Mean Amount:       ${amount_stats['legit_mean']:.2f} (Max: ${amount_stats['legit_max']:.2f})")
    print("=" * 45)

    # 6. Save cleaned data if path provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"Cleaned dataset exported to: {output_path}")

    return df, stats

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
    if RAW_DATA_PATH.exists():
        df_raw = pd.read_csv(RAW_DATA_PATH)
        clean_transaction_data(df_raw, output_path=str(PROCESSED_DATA_PATH))
    else:
        print(f"File not found: {RAW_DATA_PATH}")
