from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions with trained house price model.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.MODEL_PATH),
        help="Path to saved model artifact (.joblib).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV for prediction.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.REPORTS_DIR / "predictions.csv"),
        help="Path for output CSV with prediction column.",
    )
    return parser.parse_args()


def load_model_artifact(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    artifact = joblib.load(path)

    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], artifact

    if hasattr(artifact, "predict"):
        # Backward-compatible fallback if only pipeline object is stored.
        return artifact, {"feature_columns": None}

    raise ValueError(
        "Invalid model artifact format. Expected a dict with key 'pipeline' or a sklearn pipeline/model."
    )


def validate_input_columns(input_df: pd.DataFrame, required_columns):
    if required_columns is None:
        return

    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:10])
        raise ValueError(
            "Input CSV is missing required feature columns from training. "
            f"Missing ({len(missing_columns)}): {missing_preview}"
        )


def main() -> None:
    args = parse_args()

    print(f"[1/4] Loading model: {args.model}")
    pipeline, metadata = load_model_artifact(args.model)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[2/4] Loading input data: {input_path}")
    input_df = pd.read_csv(input_path)
    if input_df.empty:
        raise ValueError("Input CSV is empty. Please provide at least one row.")

    required_columns = metadata.get("feature_columns")
    validate_input_columns(input_df, required_columns)

    if required_columns:
        X = input_df[required_columns].copy()
    else:
        X = input_df.copy()

    print("[3/4] Running prediction")
    predictions = pipeline.predict(X)

    output_df = input_df.copy()
    output_df["prediction"] = predictions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"[4/4] Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
