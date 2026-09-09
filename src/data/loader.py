"""
Data Loader Module
Handles loading transaction data from raw CSV, processed CSV, or fallback sample files.
"""

from typing import Optional
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path

# Add project root to path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLE_DATA_PATH
from data.sample.generate_sample import generate_sample_dataset

def load_transaction_data(
    source: str = "auto",  # 'raw', 'processed', 'sample', or 'auto'
    nrows: Optional[int] = None,
    clean: bool = False
) -> pd.DataFrame:
    """
    Loads transaction dataset with automatic fallbacks.
    'auto' checks processed -> raw -> sample -> generate synthetic.
    """
    target_path = None
    
    if source == "processed":
        target_path = PROCESSED_DATA_PATH
    elif source == "raw":
        target_path = RAW_DATA_PATH
    elif source == "sample":
        target_path = SAMPLE_DATA_PATH
    elif source == "auto":
        if PROCESSED_DATA_PATH.exists():
            target_path = PROCESSED_DATA_PATH
        elif RAW_DATA_PATH.exists():
            target_path = RAW_DATA_PATH
        elif SAMPLE_DATA_PATH.exists():
            target_path = SAMPLE_DATA_PATH
        else:
            print("No existing dataset found. Generating synthetic sample dataset...")
            return generate_sample_dataset(n_samples=5000, output_path=str(SAMPLE_DATA_PATH))

    if target_path and target_path.exists():
        print(f"Loading transactions from: {target_path} (nrows={nrows})")
        df = pd.read_csv(target_path, nrows=nrows)
    else:
        print(f"Target path {target_path} not found. Falling back to sample dataset.")
        if not SAMPLE_DATA_PATH.exists():
            generate_sample_dataset(n_samples=5000, output_path=str(SAMPLE_DATA_PATH))
        df = pd.read_csv(SAMPLE_DATA_PATH, nrows=nrows)

    if clean and source != "processed":
        from src.data.cleaner import clean_transaction_data
        df, _ = clean_transaction_data(df)

    return df

if __name__ == "__main__":
    df = load_transaction_data("auto", nrows=100)
    print("Loaded shape:", df.shape)
