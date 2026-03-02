from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R2 metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, float]]:
    """Evaluate model on both train and test splits."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train": calculate_regression_metrics(y_train, train_pred),
        "test": calculate_regression_metrics(y_test, test_pred),
    }


def to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas objects into JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Index):
        return obj.tolist()
    return obj


def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
    """Save dictionary as pretty JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)
