"""
Configuration module for the Fraud Detection System.
Loads environment variables and sets path constants and business thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_transactions.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "sample_transactions.csv"

# Models directory
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Database Configuration
# Default to SQLite for seamless local execution; configurable via DATABASE_URL
DEFAULT_SQLITE_PATH = DATA_DIR / "fraud_detection.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}")

# Streaming Configuration
USE_KAFKA = os.getenv("USE_KAFKA", "False").lower() in ("true", "1", "t", "yes")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions_stream")
STREAM_DELAY = float(os.getenv("STREAM_DELAY", "0.3"))  # seconds between transactions

# Fraud Detection & Alert Thresholds
DEFAULT_FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.50"))

# Configurable Risk Tiers based on estimated fraud probability
RISK_LEVEL_THRESHOLDS = {
    "LOW": 0.30,       # prob < 0.30 -> LOW
    "MEDIUM": 0.60,    # 0.30 <= prob < 0.60 -> MEDIUM
    "HIGH": 0.85,      # 0.60 <= prob < 0.85 -> HIGH
    "CRITICAL": 1.00   # prob >= 0.85 -> CRITICAL
}

def get_risk_level(prob: float) -> str:
    """Categorize fraud probability into a standardized business risk tier."""
    if prob < RISK_LEVEL_THRESHOLDS["LOW"]:
        return "LOW"
    elif prob < RISK_LEVEL_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    elif prob < RISK_LEVEL_THRESHOLDS["HIGH"]:
        return "HIGH"
    else:
        return "CRITICAL"

# Feature definitions
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour", "day", "time_of_day_enc", "amount_bucket_enc"]
BASE_RAW_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
