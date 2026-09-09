"""
Unit tests for Feature Engineering
"""

import pytest
import numpy as np
import pandas as pd
from src.features.feature_engineering import FraudFeatureEngineer, MODEL_FEATURE_COLUMNS

def test_feature_engineering_fit_transform(sample_small_dataframe):
    fe = FraudFeatureEngineer(scaler_type="robust")
    transformed_df = fe.fit_transform(sample_small_dataframe)

    # Check derived columns exist
    for col in ["hour", "day", "time_of_day_bin", "amount_bucket_bin", "scaled_amount"]:
        assert col in transformed_df.columns

    # Verify bounds
    assert transformed_df["hour"].between(0, 23).all()
    assert transformed_df["time_of_day_bin"].isin([0, 1, 2, 3]).all()
    assert transformed_df["amount_bucket_bin"].isin([0, 1, 2, 3]).all()

def test_transform_single_transaction(sample_small_dataframe, sample_valid_transaction_dict):
    fe = FraudFeatureEngineer(scaler_type="robust")
    fe.fit(sample_small_dataframe)

    vector = fe.transform_single_transaction(sample_valid_transaction_dict)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1, len(MODEL_FEATURE_COLUMNS))
