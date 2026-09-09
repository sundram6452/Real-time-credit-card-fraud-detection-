"""
Sample Data Generator
Generates realistic synthetic credit card transactions matching the Kaggle Credit Card Fraud dataset schema.
Used for quick testing, CI/CD, and environments where raw data is omitted.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_sample_dataset(
    n_samples: int = 5000,
    fraud_ratio: float = 0.02,  # slightly higher for small test samples so tests have enough frauds
    random_state: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """Generate synthetic credit card fraud transactions."""
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # 1. Time feature: simulated seconds over 2 days (0 to 172800)
    time_legit = np.sort(np.random.uniform(0, 172800, n_legit))
    time_fraud = np.sort(np.random.uniform(0, 172800, n_fraud))
    
    # 2. PCA features V1..V28
    # Legitimate: centered around 0 with standard normal
    V_legit = np.random.normal(0, 1, size=(n_legit, 28))
    
    # Fraudulent: shifted distributions on known sensitive PCA components (V14, V10, V12, V17, V4, V11)
    V_fraud = np.random.normal(0, 1.2, size=(n_fraud, 28))
    V_fraud[:, 3] += np.random.normal(3.5, 1.0, n_fraud)    # V4 positively shifted
    V_fraud[:, 9] -= np.random.normal(4.0, 1.0, n_fraud)    # V10 negatively shifted
    V_fraud[:, 10] += np.random.normal(3.0, 1.0, n_fraud)   # V11 positively shifted
    V_fraud[:, 11] -= np.random.normal(5.0, 1.0, n_fraud)   # V12 negatively shifted
    V_fraud[:, 13] -= np.random.normal(6.0, 1.2, n_fraud)   # V14 strongly negative
    V_fraud[:, 16] -= np.random.normal(4.5, 1.2, n_fraud)   # V17 negatively shifted
    
    # 3. Amount feature
    # Legitimate: right-skewed lognormal (median ~$30, max ~$2500)
    amount_legit = np.random.lognormal(mean=3.2, sigma=1.2, size=n_legit).round(2)
    # Fraudulent: mix of small trial charges and larger unauthorized amounts
    amount_fraud = np.concatenate([
        np.random.uniform(1.0, 15.0, size=int(n_fraud * 0.4)),
        np.random.lognormal(mean=4.5, sigma=1.0, size=n_fraud - int(n_fraud * 0.4))
    ]).round(2)
    np.random.shuffle(amount_fraud)
    
    # Assemble DataFrames
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    df_legit = pd.DataFrame(
        np.column_stack([time_legit, V_legit, amount_legit, np.zeros(n_legit)]),
        columns=cols
    )
    df_fraud = pd.DataFrame(
        np.column_stack([time_fraud, V_fraud, amount_fraud, np.ones(n_fraud)]),
        columns=cols
    )
    
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['Class'] = df['Class'].astype(int)
    
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        print(f"Generated {len(df)} sample transactions saved to {output_path}")
        print(f"Fraud count: {df['Class'].sum()} ({df['Class'].mean() * 100:.2f}%)")
        
    return df

if __name__ == "__main__":
    generate_sample_dataset(
        n_samples=5000,
        fraud_ratio=0.02,
        output_path="data/sample/sample_transactions.csv"
    )
