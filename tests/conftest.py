"""
Pytest Test Fixtures and Shared Configuration
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_valid_transaction_dict():
    """Returns a valid single transaction dictionary."""
    tx = {
        "transaction_id": "tx_test_001",
        "Time": 3600.0,
        "Amount": 125.50,
        "Class": 0
    }
    for i in range(1, 29):
        tx[f"V{i}"] = float(np.random.normal(0, 1))
    return tx

@pytest.fixture
def sample_fraud_transaction_dict():
    """Returns a verified high-risk fraudulent transaction dictionary."""
    return {
        "transaction_id": "tx_fraud_test",
        "Time": 7672.0,
        "V1": 0.7027, "V2": 2.4264, "V3": -5.2345, "V4": 4.4167,
        "V5": -2.1708, "V6": -2.6676, "V7": -3.8781, "V8": 0.9113,
        "V9": -0.1662, "V10": -5.0092, "V11": 4.6757, "V12": -8.1672,
        "V13": 0.6386, "V14": -6.7633, "V15": 1.2969, "V16": -3.8118,
        "V17": -3.7541, "V18": -1.0492, "V19": 1.5542, "V20": 0.4227,
        "V21": 0.5512, "V22": -0.0098, "V23": 0.7217, "V24": 0.4732,
        "V25": -1.9593, "V26": 0.3195, "V27": 0.6005, "V28": 0.1293,
        "Amount": 1.0,
        "Class": 1
    }

@pytest.fixture
def sample_small_dataframe():
    """Returns a small pandas DataFrame with both legit and fraud transactions."""
    from data.sample.generate_sample import generate_sample_dataset
    return generate_sample_dataset(n_samples=50, fraud_ratio=0.1, random_state=42)
