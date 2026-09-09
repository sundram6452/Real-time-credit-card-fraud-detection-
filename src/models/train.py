"""
Model Training Pipeline
Trains Logistic Regression (baseline) and Random Forest (main model).
Handles severe class imbalance, performs threshold optimization,
and serializes models and evaluation metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.config import (
    MODEL_PATH,
    BASELINE_MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    DEFAULT_FRAUD_THRESHOLD,
    MODELS_DIR
)
from src.data.loader import load_transaction_data
from src.features.feature_engineering import FraudFeatureEngineer, MODEL_FEATURE_COLUMNS
from src.models.evaluate import (
    evaluate_predictions,
    analyze_threshold_curve,
    format_evaluation_summary
)
from src.database.queries import save_model_metrics

def train_fraud_detection_models(
    data_source: str = "auto",
    test_size: float = 0.20,
    random_state: int = 42,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Executes end-to-end model training, comparative evaluation, and artifact serialization.
    """
    print("=" * 55)
    print("      STARTING FRAUD DETECTION MODEL TRAINING")
    print("=" * 55)
    
    # 1. Load dataset
    df = load_transaction_data(source=data_source, clean=True)
    print(f"Total dataset records: {len(df):,}")
    print(f"Fraud count: {df['Class'].sum():,} ({df['Class'].mean() * 100:.4f}%)")

    # 2. Stratified train/test split
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"Train split: {len(X_train_raw):,} rows (Frauds: {y_train.sum()})")
    print(f"Test split:  {len(X_test_raw):,} rows (Frauds: {y_test.sum()})")

    # 3. Fit Feature Engineer on train set only (prevent data leakage)
    print("\nFitting feature engineer on training data...")
    fe = FraudFeatureEngineer(scaler_type="robust")
    X_train_trans = fe.fit_transform(X_train_raw)
    X_test_trans = fe.transform(X_test_raw)

    X_train_features = X_train_trans[MODEL_FEATURE_COLUMNS].values
    X_test_features = X_test_trans[MODEL_FEATURE_COLUMNS].values

    # 4. Train Baseline: Logistic Regression with balanced class weights
    print("\nTraining Baseline: Logistic Regression (class_weight='balanced')...")
    baseline_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state
    )
    baseline_clf.fit(X_train_features, y_train)
    y_prob_baseline = baseline_clf.predict_proba(X_test_features)[:, 1]
    baseline_metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=y_prob_baseline,
        threshold=DEFAULT_FRAUD_THRESHOLD,
        model_name="Logistic Regression (Baseline)"
    )
    print(format_evaluation_summary(baseline_metrics))

    # 5. Train Main Model: Random Forest Classifier
    print("\nTraining Main Model: Random Forest Classifier (class_weight='balanced_subsample')...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state
    )
    rf_clf.fit(X_train_features, y_train)
    y_prob_rf = rf_clf.predict_proba(X_test_features)[:, 1]
    rf_metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=y_prob_rf,
        threshold=DEFAULT_FRAUD_THRESHOLD,
        model_name="Random Forest (Main)"
    )
    print(format_evaluation_summary(rf_metrics))

    # 6. Threshold Sensitivity Analysis & Optimal Tuning
    threshold_df = analyze_threshold_curve(y_test, y_prob_rf)
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    optimal_threshold = float(best_row["threshold"])
    print(f"\nOptimal F1 Threshold for Random Forest: {optimal_threshold} (F1 = {best_row['f1']:.4f})")
    print(f"At this threshold -> Precision: {best_row['precision']:.4f}, Recall: {best_row['recall']:.4f}")
    print(f"False Positives: {int(best_row['false_positives'])}, False Negatives: {int(best_row['false_negatives'])}")

    # 7. Extract Feature Importances
    importances = rf_clf.feature_importances_
    feat_imp_list = [
        {"feature": name, "importance": float(imp)}
        for name, imp in sorted(zip(MODEL_FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    ]
    print("\nTop 10 Most Predictive Features:")
    for item in feat_imp_list[:10]:
        print(f"  {item['feature']:<20}: {item['importance']:.4f}")

    # 8. Save Artifacts
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_clf, MODEL_PATH)
        joblib.dump(baseline_clf, BASELINE_MODEL_PATH)
        fe.save(str(SCALER_PATH))

        metadata = {
            "model_type": "RandomForestClassifier",
            "version": "1.0.0",
            "test_size": test_size,
            "random_state": random_state,
            "optimal_threshold": optimal_threshold,
            "default_threshold": DEFAULT_FRAUD_THRESHOLD,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "baseline_metrics": baseline_metrics,
            "main_metrics": rf_metrics,
            "threshold_analysis": threshold_df.to_dict(orient="records"),
            "feature_importances": feat_imp_list,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "test_fraud_count": int(y_test.sum())
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nArtifacts successfully persisted:")
        print(f"  - Model:     {MODEL_PATH}")
        print(f"  - Baseline:  {BASELINE_MODEL_PATH}")
        print(f"  - Scaler:    {SCALER_PATH}")
        print(f"  - Metadata:  {METADATA_PATH}")

        # Save metrics to DB
        try:
            save_model_metrics(baseline_metrics)
            save_model_metrics(rf_metrics)
            print("Model metrics logged to database 'model_metrics' table.")
        except Exception as e:
            print(f"Warning: Could not log metrics to database: {e}")

    return {
        "baseline_model": baseline_clf,
        "main_model": rf_clf,
        "feature_engineer": fe,
        "baseline_metrics": baseline_metrics,
        "main_metrics": rf_metrics,
        "optimal_threshold": optimal_threshold,
        "feature_importances": feat_imp_list
    }

if __name__ == "__main__":
    train_fraud_detection_models(data_source="processed")
