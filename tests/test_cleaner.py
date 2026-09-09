"""
Unit tests for Data Cleaner
"""

import pytest
import pandas as pd
from src.data.cleaner import clean_transaction_data

def test_cleaner_deduplication(sample_small_dataframe):
    # Inject exact duplicates
    df_with_dups = pd.concat([sample_small_dataframe, sample_small_dataframe.iloc[:5]], ignore_index=True)
    assert len(df_with_dups) == len(sample_small_dataframe) + 5

    df_cleaned, stats = clean_transaction_data(df_with_dups, remove_duplicates=True)
    assert stats["duplicate_rows"] == 5
    assert len(df_cleaned) == len(sample_small_dataframe)
    assert stats["cleaned_rows"] == len(sample_small_dataframe)

def test_cleaner_preserves_fraud_outliers(sample_small_dataframe):
    # High-value fraud record
    df_clean, stats = clean_transaction_data(sample_small_dataframe)
    assert stats["fraud_transactions"] > 0
    assert stats["legitimate_transactions"] > 0
    assert "amount_stats" in stats
    assert stats["amount_stats"]["max"] >= stats["amount_stats"]["mean"]
