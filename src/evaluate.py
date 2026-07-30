"""Classification evaluation helpers."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the main binary-classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
