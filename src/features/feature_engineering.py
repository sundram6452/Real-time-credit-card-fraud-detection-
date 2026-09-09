"""
Feature Engineering Pipeline
Derives temporal and financial features from transaction data.
Ensures zero data leakage by strictly separating training fit from transform.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
from pathlib import Path
from src.config import SCALER_PATH

TIME_BUCKET_BINS = [-1, 6, 12, 18, 24]
TIME_BUCKET_LABELS = [0, 1, 2, 3]  # 0: Night, 1: Morning, 2: Afternoon, 3: Evening
TIME_BUCKET_NAMES = ["Night (00-06)", "Morning (06-12)", "Afternoon (12-18)", "Evening (18-24)"]

AMOUNT_BUCKET_BINS = [-1.0, 20.0, 100.0, 500.0, float("inf")]
AMOUNT_BUCKET_LABELS = [0, 1, 2, 3]  # 0: Low, 1: Medium, 2: High, 3: Very High
AMOUNT_BUCKET_NAMES = ["Low ($0-$20)", "Medium ($20-$100)", "High ($100-$500)", "Very High ($500+)"]

MODEL_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "scaled_amount",
    "hour",
    "day",
    "time_of_day_bin",
    "amount_bucket_bin"
]

class FraudFeatureEngineer:
    """
    Stateful feature engineering class that fits scalers on training data
    and applies identical transformations to test and real-time streaming data.
    """
    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.is_fitted = False

    def _derive_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives hour, day, and time-of-day bins from Time (seconds)."""
        time_series = df["Time"]
        hours = ((time_series // 3600) % 24).astype(int)
        days = ((time_series // 86400) + 1).astype(int)
        time_bins = pd.cut(hours, bins=TIME_BUCKET_BINS, labels=TIME_BUCKET_LABELS).astype(int)
        
        df = df.copy()
        df["hour"] = hours
        df["day"] = days
        df["time_of_day_bin"] = time_bins
        return df

    def _derive_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives amount brackets from Amount ($)."""
        amounts = df["Amount"]
        amt_bins = pd.cut(amounts, bins=AMOUNT_BUCKET_BINS, labels=AMOUNT_BUCKET_LABELS).astype(int)
        df = df.copy()
        df["amount_bucket_bin"] = amt_bins
        return df

    def fit(self, df: pd.DataFrame) -> "FraudFeatureEngineer":
        """Fits the scaler strictly on the training set Amount feature."""
        self.scaler.fit(df[["Amount"]])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms DataFrame by adding derived features and scaling Amount."""
        if not self.is_fitted:
            raise ValueError("FraudFeatureEngineer must be fitted before transforming data.")
        
        df_trans = self._derive_temporal_features(df)
        df_trans = self._derive_amount_features(df_trans)
        df_trans["scaled_amount"] = self.scaler.transform(df_trans[["Amount"]]).ravel()
        return df_trans

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit and transform training data."""
        return self.fit(df).transform(df)

    def transform_single_transaction(self, tx: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single transaction dict into the model's exact feature vector.
        Optimized for real-time low-latency inference.
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet.")

        t = float(tx["Time"])
        amt = float(tx["Amount"])
        
        # Temporal
        hour = int((t // 3600) % 24)
        day = int((t // 86400) + 1)
        if hour <= 6:
            time_bin = 0
        elif hour <= 12:
            time_bin = 1
        elif hour <= 18:
            time_bin = 2
        else:
            time_bin = 3

        # Amount bucket
        if amt <= 20.0:
            amt_bin = 0
        elif amt <= 100.0:
            amt_bin = 1
        elif amt <= 500.0:
            amt_bin = 2
        else:
            amt_bin = 3

        # Scaled amount
        amt_df = pd.DataFrame([[amt]], columns=["Amount"])
        scaled_amt = float(self.scaler.transform(amt_df)[0][0])

        # Assemble feature row matching MODEL_FEATURE_COLUMNS:
        # V1..V28, scaled_amount, hour, day, time_of_day_bin, amount_bucket_bin
        v_features = [float(tx[f"V{i}"]) for i in range(1, 29)]
        feature_row = v_features + [scaled_amt, hour, day, time_bin, amt_bin]
        return np.array([feature_row], dtype=np.float32)

    def save(self, path: Optional[str] = None):
        """Persists the feature engineer/scaler artifact."""
        save_path = Path(path) if path else SCALER_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        print(f"Feature engineer saved to: {save_path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "FraudFeatureEngineer":
        """Loads a persisted feature engineer artifact."""
        load_path = Path(path) if path else SCALER_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"Feature engineer artifact not found at: {load_path}")
        return joblib.load(load_path)
