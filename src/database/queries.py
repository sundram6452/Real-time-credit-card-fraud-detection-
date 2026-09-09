"""
Database Query Utilities & Persistence Helpers
Executes business queries and persists streaming transactions, predictions, and alerts.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import text
from src.database.connection import (
    get_engine,
    get_session,
    Transaction,
    FraudPrediction,
    FraudAlert,
    ModelMetric
)

def save_transaction(tx_data: Dict[str, Any], session=None) -> str:
    """Inserts a single transaction record into the database."""
    should_close = False
    if session is None:
        session = get_session()
        should_close = True

    tx_id = str(tx_data.get("transaction_id", f"tx_{uuid.uuid4().hex[:12]}"))
    
    tx = Transaction(
        transaction_id=tx_id,
        time=float(tx_data["Time"]),
        v1=float(tx_data["V1"]),
        v2=float(tx_data["V2"]),
        v3=float(tx_data["V3"]),
        v4=float(tx_data["V4"]),
        v5=float(tx_data["V5"]),
        v6=float(tx_data["V6"]),
        v7=float(tx_data["V7"]),
        v8=float(tx_data["V8"]),
        v9=float(tx_data["V9"]),
        v10=float(tx_data["V10"]),
        v11=float(tx_data["V11"]),
        v12=float(tx_data["V12"]),
        v13=float(tx_data["V13"]),
        v14=float(tx_data["V14"]),
        v15=float(tx_data["V15"]),
        v16=float(tx_data["V16"]),
        v17=float(tx_data["V17"]),
        v18=float(tx_data["V18"]),
        v19=float(tx_data["V19"]),
        v20=float(tx_data["V20"]),
        v21=float(tx_data["V21"]),
        v22=float(tx_data["V22"]),
        v23=float(tx_data["V23"]),
        v24=float(tx_data["V24"]),
        v25=float(tx_data["V25"]),
        v26=float(tx_data["V26"]),
        v27=float(tx_data["V27"]),
        v28=float(tx_data["V28"]),
        amount=float(tx_data["Amount"]),
        class_label=int(tx_data.get("Class", 0))
    )

    try:
        session.merge(tx)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if should_close:
            session.close()

    return tx_id

def save_prediction(pred_data: Dict[str, Any], session=None) -> str:
    """Inserts a fraud model prediction into the database."""
    should_close = False
    if session is None:
        session = get_session()
        should_close = True

    pred_id = str(pred_data.get("prediction_id", f"pred_{uuid.uuid4().hex[:12]}"))
    
    pred = FraudPrediction(
        prediction_id=pred_id,
        transaction_id=str(pred_data["transaction_id"]),
        fraud_probability=float(pred_data["fraud_probability"]),
        prediction=int(pred_data["prediction"]),
        threshold_used=float(pred_data.get("threshold_used", 0.50)),
        model_version=str(pred_data.get("model_version", "v1.0"))
    )

    try:
        session.add(pred)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if should_close:
            session.close()

    return pred_id

def save_alert(alert_data: Dict[str, Any], session=None) -> str:
    """Inserts an operational fraud alert into the database."""
    should_close = False
    if session is None:
        session = get_session()
        should_close = True

    alert_id = str(alert_data.get("alert_id", f"alt_{uuid.uuid4().hex[:12]}"))
    
    alert = FraudAlert(
        alert_id=alert_id,
        transaction_id=str(alert_data["transaction_id"]),
        amount=float(alert_data["amount"]),
        fraud_probability=float(alert_data["fraud_probability"]),
        risk_level=str(alert_data["risk_level"]),
        alert_status=str(alert_data.get("alert_status", "NEW"))
    )

    try:
        session.add(alert)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if should_close:
            session.close()

    return alert_id

def save_model_metrics(metrics: Dict[str, Any], session=None) -> str:
    """Saves model evaluation metrics to the database."""
    should_close = False
    if session is None:
        session = get_session()
        should_close = True

    metric_id = str(uuid.uuid4().hex[:12])
    m = ModelMetric(
        metric_id=metric_id,
        model_name=str(metrics["model_name"]),
        accuracy=float(metrics["accuracy"]),
        precision_score=float(metrics["precision"]),
        recall_score=float(metrics["recall"]),
        f1_score=float(metrics["f1"]),
        roc_auc=float(metrics["roc_auc"]),
        pr_auc=float(metrics["pr_auc"])
    )

    try:
        session.add(m)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if should_close:
            session.close()

    return metric_id

def run_raw_sql(sql: str) -> pd.DataFrame:
    """Executes any raw SQL query and returns the results as a pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df

def get_recent_alerts(limit: int = 50) -> pd.DataFrame:
    """Retrieves the most recent alerts ordered by timestamp descending."""
    sql = f"""
    SELECT 
        a.alert_id,
        a.transaction_id,
        a.amount,
        a.fraud_probability,
        a.risk_level,
        a.alert_status,
        a.timestamp
    FROM fraud_alerts a
    ORDER BY a.timestamp DESC
    LIMIT {limit};
    """
    return run_raw_sql(sql)

def get_recent_transactions(limit: int = 50) -> pd.DataFrame:
    """Retrieves the most recent transactions with their model predictions."""
    sql = f"""
    SELECT 
        t.transaction_id,
        t.time,
        t.amount,
        t.class AS actual_class,
        p.fraud_probability,
        p.prediction,
        p.predicted_at
    FROM transactions t
    LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
    ORDER BY t.created_at DESC
    LIMIT {limit};
    """
    return run_raw_sql(sql)

def populate_database_from_csv(csv_path: str, max_rows: Optional[int] = 10000, batch_size: int = 1000):
    """
    Batched bulk ingestion from CSV into the relational database.
    Useful for initializing analytics tables with historical transactions.
    """
    from src.database.connection import init_db
    init_db()
    
    print(f"Ingesting transactions from {csv_path} into database (max_rows={max_rows})...")
    df = pd.read_csv(csv_path, nrows=max_rows)
    
    # Format columns matching table
    df_db = df.copy()
    if "transaction_id" not in df_db.columns:
        df_db["transaction_id"] = [f"tx_{i:06d}" for i in range(len(df_db))]
    
    # Rename Class to class if needed
    col_map = {c: c.lower() for c in df_db.columns}
    col_map["Class"] = "class"
    df_db = df_db.rename(columns=col_map)
    
    engine = get_engine()
    df_db.to_sql("transactions", con=engine, if_exists="append", index=False, chunksize=batch_size)
    print(f"Successfully loaded {len(df_db)} transactions into database.")
