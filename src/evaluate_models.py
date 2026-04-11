"""
Метрики качества, ROC, матрица ошибок, сводная таблица по моделям.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Считаем основные метрики для одной модели."""
    y_pred = model.predict(X_test)
    # вероятность положительного класса (дефолт)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred.astype(float)

    out = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "y_pred": y_pred,
        "y_score": y_score,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": None,
        "tpr": None,
    }
    fpr, tpr, _ = roc_curve(y_test, y_score)
    out["fpr"] = fpr
    out["tpr"] = tpr
    return out


def metrics_to_row(result: Dict[str, Any]) -> dict:
    return {
        "model": result["model"],
        "accuracy": round(result["accuracy"], 4),
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "f1": round(result["f1"], 4),
        "roc_auc": round(result["roc_auc"], 4),
    }


def evaluate_all_models(
    fitted_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    detailed = {}
    for name, model in fitted_models.items():
        res = evaluate_binary_classifier(model, X_test, y_test, name)
        detailed[name] = res
        rows.append(metrics_to_row(res))
    return pd.DataFrame(rows), detailed


def save_metrics_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
