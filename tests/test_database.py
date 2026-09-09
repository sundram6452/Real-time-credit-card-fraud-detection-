"""
Unit tests for Database Connection and Persistence
"""

import pytest
from src.database.connection import init_db, get_session
from src.database.queries import (
    save_transaction,
    save_prediction,
    save_alert,
    run_raw_sql
)

def test_database_persistence(sample_valid_transaction_dict):
    init_db()
    
    # 1. Save transaction
    tx_id = save_transaction(sample_valid_transaction_dict)
    assert tx_id is not None

    # 2. Save prediction
    pred_id = save_prediction({
        "transaction_id": tx_id,
        "fraud_probability": 0.05,
        "prediction": 0,
        "threshold_used": 0.50,
        "model_version": "v1.0"
    })
    assert pred_id is not None

    # 3. Save alert
    alert_id = save_alert({
        "transaction_id": tx_id,
        "amount": sample_valid_transaction_dict["Amount"],
        "fraud_probability": 0.95,
        "risk_level": "CRITICAL"
    })
    assert alert_id is not None

    # 4. Verify SQL query
    res = run_raw_sql(f"SELECT transaction_id, amount FROM transactions WHERE transaction_id = '{tx_id}'")
    assert len(res) == 1
    assert float(res.iloc[0]["amount"]) == sample_valid_transaction_dict["Amount"]
