"""
Inference Pipeline Module
Loads serialized model artifacts and performs low-latency predictions
on single transactions or micro-batches.
"""

from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    DEFAULT_FRAUD_THRESHOLD,
    get_risk_level
)
from src.features.feature_engineering import FraudFeatureEngineer, MODEL_FEATURE_COLUMNS

_model = None
_feature_engineer = None
_metadata = None

def get_loaded_model(model_path: Optional[str] = None):
    """Singleton getter for the trained model."""
    global _model
    if _model is None:
        p = Path(model_path) if model_path else MODEL_PATH
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found at: {p}. Run train.py first.")
        _model = joblib.load(p)
    return _model

def get_loaded_feature_engineer(scaler_path: Optional[str] = None):
    """Singleton getter for the feature engineer / scaler."""
    global _feature_engineer
    if _feature_engineer is None:
        p = Path(scaler_path) if scaler_path else SCALER_PATH
        if not p.exists():
            raise FileNotFoundError(f"Scaler artifact not found at: {p}. Run train.py first.")
        _feature_engineer = FraudFeatureEngineer.load(str(p))
    return _feature_engineer

def get_model_metadata(metadata_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads model performance and threshold metadata."""
    global _metadata
    if _metadata is None:
        p = Path(metadata_path) if metadata_path else METADATA_PATH
        if p.exists():
            with open(p, "r") as f:
                _metadata = json.load(f)
        else:
            _metadata = {
                "model_type": "RandomForestClassifier",
                "version": "1.0.0",
                "optimal_threshold": DEFAULT_FRAUD_THRESHOLD,
                "default_threshold": DEFAULT_FRAUD_THRESHOLD
            }
    return _metadata

def predict_single_transaction(
    tx: Dict[str, Any],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates a single transaction dictionary in real time.
    Returns:
      - transaction_id
      - amount
      - fraud_probability (float 0.0 - 1.0)
      - prediction (0 for Legit, 1 for Fraud)
      - risk_level (LOW, MEDIUM, HIGH, CRITICAL)
      - action_recommendation (APPROVE, REVIEW, DECLINE)
    """
    model = get_loaded_model()
    fe = get_loaded_feature_engineer()
    metadata = get_model_metadata()

    effective_threshold = (
        threshold
        if threshold is not None
        else metadata.get("optimal_threshold", DEFAULT_FRAUD_THRESHOLD)
    )

    # 1. Transform features
    feat_vector = fe.transform_single_transaction(tx)

    # 2. Predict probability
    prob = float(model.predict_proba(feat_vector)[0][1])
    is_fraud = int(prob >= effective_threshold)
    risk_level = get_risk_level(prob)

    # 3. Business action recommendation
    if risk_level == "LOW":
        recommendation = "APPROVE"
    elif risk_level == "MEDIUM":
        recommendation = "FLAG_FOR_MONITORING"
    elif risk_level == "HIGH":
        recommendation = "REQUIRE_MANUAL_REVIEW"
    else:  # CRITICAL
        recommendation = "DECLINE_AND_ALERT"

    tx_id = tx.get("transaction_id", "tx_unknown")
    amount = float(tx.get("Amount", 0.0))

    return {
        "transaction_id": tx_id,
        "amount": amount,
        "fraud_probability": round(prob, 4),
        "prediction": is_fraud,
        "is_fraud": bool(is_fraud),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "threshold_used": effective_threshold,
        "model_version": metadata.get("version", "1.0.0")
    }

def predict_batch(
    df: pd.DataFrame,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """Batch prediction on an entire DataFrame."""
    model = get_loaded_model()
    fe = get_loaded_feature_engineer()
    metadata = get_model_metadata()

    effective_threshold = (
        threshold
        if threshold is not None
        else metadata.get("optimal_threshold", DEFAULT_FRAUD_THRESHOLD)
    )

    df_trans = fe.transform(df)
    X = df_trans[MODEL_FEATURE_COLUMNS].values
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= effective_threshold).astype(int)

    df_out = df.copy()
    df_out["fraud_probability"] = probs.round(4)
    df_out["prediction"] = preds
    df_out["risk_level"] = [get_risk_level(p) for p in probs]
    return df_out
