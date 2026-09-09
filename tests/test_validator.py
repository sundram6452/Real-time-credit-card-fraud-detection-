"""
Unit tests for Data Validator
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validator import validate_transaction_dict, validate_dataframe

def test_validate_valid_transaction_dict(sample_valid_transaction_dict):
    is_valid, msg = validate_transaction_dict(sample_valid_transaction_dict)
    assert is_valid is True
    assert msg == "Valid"

def test_validate_missing_column(sample_valid_transaction_dict):
    del sample_valid_transaction_dict["Amount"]
    is_valid, msg = validate_transaction_dict(sample_valid_transaction_dict)
    assert is_valid is False
    assert "Missing required column: Amount" in msg

def test_validate_negative_amount(sample_valid_transaction_dict):
    sample_valid_transaction_dict["Amount"] = -15.00
    is_valid, msg = validate_transaction_dict(sample_valid_transaction_dict)
    assert is_valid is False
    assert "non-negative" in msg

def test_validate_nan_value(sample_valid_transaction_dict):
    sample_valid_transaction_dict["V14"] = float("nan")
    is_valid, msg = validate_transaction_dict(sample_valid_transaction_dict)
    assert is_valid is False
    assert "cannot be null or NaN" in msg

def test_validate_dataframe_valid(sample_small_dataframe):
    report = validate_dataframe(sample_small_dataframe)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["negative_amounts"] == 0

def test_validate_dataframe_invalid_class(sample_small_dataframe):
    df_invalid = sample_small_dataframe.copy()
    df_invalid.loc[0, "Class"] = 99
    report = validate_dataframe(df_invalid)
    assert report["is_valid"] is False
    assert report["invalid_classes"] > 0
