from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from . import config
from .predict import load_model_artifact, validate_input_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve house price model as a Flask API.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", str(config.MODEL_PATH)),
        help="Path to saved model artifact (.joblib).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for Flask app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for Flask app.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode.",
    )
    return parser.parse_args()


def _error_response(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def _extract_instances(payload: Any) -> Tuple[list[dict], str | None]:
    if payload is None:
        return [], "Request body must be valid JSON."

    if isinstance(payload, dict) and "instances" in payload:
        records = payload["instances"]
    elif isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        return [], "JSON body must be an object, a list of objects, or {'instances': [...]}."

    if not isinstance(records, list) or len(records) == 0:
        return [], "No input records found. Provide at least one input row."

    if not all(isinstance(row, dict) for row in records):
        return [], "Each input record must be a JSON object."

    return records, None


def create_app(model_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    resolved_model_path = Path(model_path or os.getenv("MODEL_PATH", str(config.MODEL_PATH)))
    model_state = {
        "pipeline": None,
        "metadata": {},
        "error": None,
    }

    try:
        pipeline, metadata = load_model_artifact(resolved_model_path)
        model_state["pipeline"] = pipeline
        model_state["metadata"] = metadata
    except Exception as exc:  # pragma: no cover - runtime health check handles this
        model_state["error"] = str(exc)

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "house-price-api",
                "status_endpoint": "/health",
                "predict_endpoint": "/predict",
            }
        )

    @app.get("/health")
    def health():
        if model_state["error"] is not None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "model_loaded": False,
                        "model_path": str(resolved_model_path),
                        "error": model_state["error"],
                    }
                ),
                500,
            )

        required_columns = model_state["metadata"].get("feature_columns") or []
        return jsonify(
            {
                "status": "ok",
                "model_loaded": True,
                "model_path": str(resolved_model_path),
                "required_feature_count": len(required_columns),
                "required_features": required_columns,
            }
        )

    @app.post("/predict")
    def predict():
        if model_state["error"] is not None:
            return _error_response(
                f"Model failed to load: {model_state['error']}",
                500,
            )

        payload = request.get_json(silent=True)
        records, extraction_error = _extract_instances(payload)
        if extraction_error is not None:
            return _error_response(extraction_error, 400)

        input_df = pd.DataFrame.from_records(records)
        required_columns = model_state["metadata"].get("feature_columns")

        try:
            validate_input_columns(input_df, required_columns)
        except ValueError as exc:
            return _error_response(str(exc), 400)

        if required_columns:
            features_df = input_df[required_columns].copy()
        else:
            features_df = input_df.copy()

        try:
            predictions = model_state["pipeline"].predict(features_df)
        except Exception as exc:
            return _error_response(f"Prediction failed: {exc}", 500)

        predictions_list = [float(x) for x in np.asarray(predictions).ravel().tolist()]
        response = {
            "count": len(predictions_list),
            "predictions": predictions_list,
        }
        if len(predictions_list) == 1:
            response["prediction"] = predictions_list[0]

        return jsonify(response)

    return app


app = create_app()


def main() -> None:
    args = parse_args()
    flask_app = create_app(model_path=args.model)
    flask_app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
