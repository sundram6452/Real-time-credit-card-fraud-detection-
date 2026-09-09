"""
Database Connection & ORM Models
Compatible with SQLite, PostgreSQL, and MySQL using SQLAlchemy.
"""

import os
from datetime import datetime
from typing import Generator
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from src.config import DATABASE_URL, BASE_DIR

Base = declarative_base()

class Transaction(Base):
    """Raw / Ingested Credit Card Transaction."""
    __tablename__ = "transactions"

    transaction_id = Column(String(64), primary_key=True)
    time = Column(Float, nullable=False, index=True)
    v1 = Column(Float, nullable=False)
    v2 = Column(Float, nullable=False)
    v3 = Column(Float, nullable=False)
    v4 = Column(Float, nullable=False)
    v5 = Column(Float, nullable=False)
    v6 = Column(Float, nullable=False)
    v7 = Column(Float, nullable=False)
    v8 = Column(Float, nullable=False)
    v9 = Column(Float, nullable=False)
    v10 = Column(Float, nullable=False)
    v11 = Column(Float, nullable=False)
    v12 = Column(Float, nullable=False)
    v13 = Column(Float, nullable=False)
    v14 = Column(Float, nullable=False)
    v15 = Column(Float, nullable=False)
    v16 = Column(Float, nullable=False)
    v17 = Column(Float, nullable=False)
    v18 = Column(Float, nullable=False)
    v19 = Column(Float, nullable=False)
    v20 = Column(Float, nullable=False)
    v21 = Column(Float, nullable=False)
    v22 = Column(Float, nullable=False)
    v23 = Column(Float, nullable=False)
    v24 = Column(Float, nullable=False)
    v25 = Column(Float, nullable=False)
    v26 = Column(Float, nullable=False)
    v27 = Column(Float, nullable=False)
    v28 = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    class_label = Column("class", Integer, default=0, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    predictions = relationship("FraudPrediction", back_populates="transaction", cascade="all, delete-orphan")
    alerts = relationship("FraudAlert", back_populates="transaction", cascade="all, delete-orphan")

class FraudPrediction(Base):
    """Model Inference Output for an Ingested Transaction."""
    __tablename__ = "fraud_predictions"

    prediction_id = Column(String(64), primary_key=True)
    transaction_id = Column(String(64), ForeignKey("transactions.transaction_id"), nullable=False, index=True)
    fraud_probability = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)
    threshold_used = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="predictions")

class FraudAlert(Base):
    """Business Alert for Suspicious or High-Risk Transactions."""
    __tablename__ = "fraud_alerts"

    alert_id = Column(String(64), primary_key=True)
    transaction_id = Column(String(64), ForeignKey("transactions.transaction_id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    alert_status = Column(String(16), default="NEW")            # NEW, REVIEWED, RESOLVED, DISMISSED
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    transaction = relationship("Transaction", back_populates="alerts")

class ModelMetric(Base):
    """Historical ML Validation & Tracking Metrics."""
    __tablename__ = "model_metrics"

    metric_id = Column(String(64), primary_key=True)
    model_name = Column(String(64), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision_score = Column(Float, nullable=False)
    recall_score = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

_engine = None
_session_factory = None

def get_engine(db_url: str = DATABASE_URL):
    """Creates and returns the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)
    return _engine

def get_session_factory(db_url: str = DATABASE_URL):
    """Returns the thread-local session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(db_url)
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _session_factory

def get_session(db_url: str = DATABASE_URL) -> Session:
    """Provides a new database session."""
    factory = get_session_factory(db_url)
    return factory()

def init_db(db_url: str = DATABASE_URL, drop_first: bool = False):
    """Initializes schema and tables in the database."""
    engine = get_engine(db_url)
    if drop_first:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print(f"Database schema initialized successfully at: {db_url}")

if __name__ == "__main__":
    init_db()
