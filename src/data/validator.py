"""
Transaction Data Validator
Implements schema and range validation for both batch DataFrames and individual streaming transactions.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def validate_transaction_dict(tx: Dict[str, Any], require_class: bool = False) -> Tuple[bool, str]:
    """
    Validates a single transaction dictionary (used in real-time streaming ingestion).
    Returns (is_valid, error_message).
    """
    if not isinstance(tx, dict):
        return False, "Transaction payload must be a JSON object / dictionary"
    
    # Check missing required fields
    for col in REQUIRED_COLUMNS:
        if col not in tx:
            return False, f"Missing required column: {col}"
        val = tx[col]
        if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
            return False, f"Column '{col}' cannot be null or NaN"
        if not isinstance(val, (int, float, np.number)):
            return False, f"Column '{col}' must be numeric, got {type(val).__name__}"
    
    # Check bounds
    if tx["Amount"] < 0:
        return False, f"Invalid Amount: {tx['Amount']}. Amount must be non-negative."
    
    if tx["Time"] < 0:
        return False, f"Invalid Time: {tx['Time']}. Time must be non-negative."
    
    if require_class and "Class" in tx:
        if tx["Class"] not in (0, 1):
            return False, f"Class label must be 0 or 1, got {tx['Class']}"
            
    return True, "Valid"

def validate_dataframe(df: pd.DataFrame, require_class: bool = True) -> Dict[str, Any]:
    """
    Performs comprehensive batch validation on a DataFrame.
    Returns a dictionary of validation metrics and flags.
    """
    results: Dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": [],
        "null_counts": {},
        "negative_amounts": 0,
        "invalid_classes": 0
    }
    
    # Check required columns
    expected_cols = REQUIRED_COLUMNS + (["Class"] if require_class else [])
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        results["is_valid"] = False
        results["missing_columns"] = missing
        results["errors"].append(f"Missing required columns: {missing}")
        return results

    # Check nulls
    null_counts = df[expected_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    results["null_counts"] = null_counts[null_counts > 0].to_dict()
    if total_nulls > 0:
        results["is_valid"] = False
        results["errors"].append(f"Found {total_nulls} null/missing values across columns.")

    # Check non-numeric types
    for col in expected_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            results["is_valid"] = False
            results["errors"].append(f"Column '{col}' is not numeric ({df[col].dtype}).")

    # Range checks
    neg_amounts = int((df["Amount"] < 0).sum())
    results["negative_amounts"] = neg_amounts
    if neg_amounts > 0:
        results["is_valid"] = False
        results["errors"].append(f"Found {neg_amounts} records with negative Amount.")

    if require_class and "Class" in df.columns:
        invalid_class_count = int((~df["Class"].isin([0, 1])).sum())
        results["invalid_classes"] = invalid_class_count
        if invalid_class_count > 0:
            results["is_valid"] = False
            results["errors"].append(f"Found {invalid_class_count} records with invalid Class label.")

    return results
