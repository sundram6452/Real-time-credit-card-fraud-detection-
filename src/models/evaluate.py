"""
Model Evaluation Module
Computes Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix,
and analyzes threshold trade-offs between false positives and false negatives.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates binary classification predictions at a specified probability threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    # Compute ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    
    # Sample down curves to at most 100 points for compact JSON serialization
    step_roc = max(1, len(fpr) // 100)
    step_pr = max(1, len(precisions) // 100)

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "roc_curve": {
            "fpr": fpr[::step_roc].tolist(),
            "tpr": tpr[::step_roc].tolist()
        },
        "pr_curve": {
            "precision": precisions[::step_pr].tolist(),
            "recall": recalls[::step_pr].tolist()
        }
    }
    return metrics

def analyze_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Sweeps probability thresholds to evaluate trade-offs between
    False Positives (legitimate transactions blocked) and
    False Negatives (fraudulent transactions missed).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        rows.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn
        })

    return pd.DataFrame(rows)

def format_evaluation_summary(metrics: Dict[str, Any]) -> str:
    """Creates a clean ASCII summary of model metrics."""
    return f"""
--------------------------------------------------
MODEL EVALUATION SUMMARY: {metrics['model_name']}
--------------------------------------------------
Classification Threshold: {metrics['threshold']}
Accuracy:                 {metrics['accuracy']:.4f}
Precision:                {metrics['precision']:.4f}
Recall:                   {metrics['recall']:.4f}
F1 Score:                 {metrics['f1']:.4f}
ROC-AUC:                  {metrics['roc_auc']:.4f}
PR-AUC:                   {metrics['pr_auc']:.4f}
--------------------------------------------------
Confusion Matrix:
  True Negatives (Legit Allowed):  {metrics['true_negatives']:,}
  False Positives (False Alarms):  {metrics['false_positives']:,}
  False Negatives (Fraud Missed):  {metrics['false_negatives']:,}
  True Positives (Fraud Caught):   {metrics['true_positives']:,}
--------------------------------------------------
"""
