from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor

from . import config
from .data import build_preprocessor, load_data, split_data
from .utils import evaluate_model, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train house price regression model.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(config.RAW_DATA_PATH),
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=config.TARGET_COLUMN,
        help="Target column name.",
    )
    return parser.parse_args()


def build_model_candidates() -> dict:
    random_forest = RandomForestRegressor(
        n_estimators=250,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=5,
        max_features="sqrt",
    )

    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "random_forest": random_forest,
        "random_forest_log_target": TransformedTargetRegressor(
            regressor=clone(random_forest),
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=config.RANDOM_STATE,
            subsample=0.8,
        ),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    config.ensure_directories()

    print(f"[1/5] Loading data: {data_path}")
    df = load_data(data_path)
    print(f"      Rows={len(df):,}, Columns={df.shape[1]:,}")

    if args.target not in df.columns:
        raise ValueError(
            "Target column not found. "
            f"Expected '{args.target}'. TODO: ganti argumen --target sesuai nama kolom target sebenarnya."
        )

    print(f"[2/5] Splitting data with target='{args.target}'")
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=args.target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    used_rows = len(X_train) + len(X_test)
    dropped_rows = len(df) - used_rows
    if dropped_rows > 0:
        print(f"      Rows used after target cleaning: {used_rows:,} (dropped={dropped_rows:,})")
    else:
        print(f"      Rows used after target cleaning: {used_rows:,}")

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    feature_columns = numeric_features + categorical_features
    print(
        "      Feature summary: "
        f"numeric={len(numeric_features)}, categorical={len(categorical_features)}"
    )

    print("[3/5] Training candidate models")
    candidates = build_model_candidates()
    results = {}
    trained_pipelines = {}

    for model_name, model in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
        results[model_name] = metrics
        trained_pipelines[model_name] = pipeline

        print(
            f"      - {model_name}: "
            f"test_rmse={metrics['test']['rmse']:.4f}, "
            f"test_mae={metrics['test']['mae']:.4f}, "
            f"test_r2={metrics['test']['r2']:.4f}"
        )

    print("[4/5] Selecting best model by lowest test RMSE")
    best_model_name = min(results, key=lambda name: results[name]["test"]["rmse"])
    best_pipeline = trained_pipelines[best_model_name]
    best_rmse = results[best_model_name]["test"]["rmse"]
    print(f"      Best model: {best_model_name} (test_rmse={best_rmse:.4f})")

    model_artifact = {
        "pipeline": best_pipeline,
        "feature_columns": feature_columns,
        "target_column": args.target,
        "best_model_name": best_model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(model_artifact, config.MODEL_PATH)

    metrics_payload = {
        "data_path": str(data_path),
        "target_column": args.target,
        "n_rows": len(df),
        "n_rows_used": used_rows,
        "n_rows_dropped_target_invalid": dropped_rows,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "models": results,
        "best_model": {
            "name": best_model_name,
            "selection_metric": "test_rmse",
            "value": best_rmse,
            "metrics": results[best_model_name],
        },
    }
    save_json(metrics_payload, config.METRICS_PATH)

    print("[5/5] Saving artifacts")
    print(f"      Model saved to: {config.MODEL_PATH}")
    print(f"      Metrics saved to: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
