"""
Unit tests for Model Predictions & Risk Tiers
"""

import pytest
from src.models.predict import predict_single_transaction, get_model_metadata

def test_predict_single_legitimate_transaction(sample_valid_transaction_dict):
    result = predict_single_transaction(sample_valid_transaction_dict)
    assert "fraud_probability" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert result["recommendation"] in ("APPROVE", "FLAG_FOR_MONITORING", "REQUIRE_MANUAL_REVIEW", "DECLINE_AND_ALERT")

def test_predict_single_fraud_transaction(sample_fraud_transaction_dict):
    result = predict_single_transaction(sample_fraud_transaction_dict)
    # Fraud profile should yield high probability
    assert result["fraud_probability"] > 0.50
    assert result["prediction"] == 1
    assert result["risk_level"] in ("HIGH", "CRITICAL")

def test_model_metadata_loaded():
    metadata = get_model_metadata()
    assert "version" in metadata
    assert "optimal_threshold" in metadata
    assert "feature_columns" in metadata
