from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load dataset from CSV and validate basic assumptions."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {path}")

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify_target: bool = True,
    n_stratify_bins: int = 10,
):
    """Split dataframe into train/test features and target."""
    if target_col not in df.columns:
        raise ValueError(
            "Target column not found in dataset. "
            f"Expected '{target_col}'. TODO: gunakan --target <nama_kolom_target_yang_benar>."
        )

    target_series = _coerce_target_to_numeric(df[target_col])
    valid_target_mask = target_series.notna()
    dropped_rows = int((~valid_target_mask).sum())

    if dropped_rows > 0:
        print(
            f"[WARN] Dropping {dropped_rows} row(s) because target '{target_col}' is missing/non-numeric."
        )

    if not valid_target_mask.any():
        raise ValueError(
            "All target values are missing or non-numeric after cleaning. "
            f"Please inspect target column '{target_col}'."
        )

    X = df.loc[valid_target_mask].drop(columns=[target_col])
    y = target_series.loc[valid_target_mask]

    if X.shape[1] == 0:
        raise ValueError("No feature columns available after removing target column.")

    stratify_values = None
    if stratify_target:
        n_unique_target = int(y.nunique(dropna=True))
        n_bins = max(2, min(n_stratify_bins, n_unique_target))
        if n_bins >= 2:
            try:
                stratify_values = pd.qcut(y, q=n_bins, duplicates="drop")
                if getattr(stratify_values, "nunique", lambda: 0)() <= 1:
                    stratify_values = None
            except ValueError:
                stratify_values = None

        if stratify_values is not None:
            print(
                f"[INFO] Using stratified split on target quantiles (bins={stratify_values.nunique()})."
            )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def _coerce_target_to_numeric(target: pd.Series) -> pd.Series:
    """
    Convert target values to numeric while tolerating strings like
    currency symbols or thousands separators.
    """
    if pd.api.types.is_numeric_dtype(target):
        return pd.to_numeric(target, errors="coerce")

    cleaned = (
        target.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _extract_first_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(_NUMBER_RE, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _parse_amount_in_rupees(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower().str.strip()
    value = _extract_first_number(text)

    multiplier = pd.Series(1.0, index=text.index)
    multiplier = multiplier.mask(text.str.contains(r"\bcrore\b|\bcr\b", regex=True, na=False), 1e7)
    multiplier = multiplier.mask(text.str.contains(r"\blakh\b|\blac\b", regex=True, na=False), 1e5)
    multiplier = multiplier.mask(text.str.contains(r"\bmillion\b|\bmn\b", regex=True, na=False), 1e6)
    multiplier = multiplier.mask(text.str.contains(r"\bk\b|\bthousand\b", regex=True, na=False), 1e3)

    return value * multiplier


def _parse_area_to_sqft(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower().str.strip()
    value = _extract_first_number(text)

    multiplier = pd.Series(1.0, index=text.index)
    multiplier = multiplier.mask(
        text.str.contains(r"\bsq\.?\s*m\b|\bsqm\b|\bm2\b|square meter", regex=True, na=False),
        10.7639,
    )
    multiplier = multiplier.mask(
        text.str.contains(r"\bsq\.?\s*yd\b|\bsqyd\b|yard", regex=True, na=False),
        9.0,
    )
    multiplier = multiplier.mask(text.str.contains(r"\bacre\b", regex=True, na=False), 43560.0)
    multiplier = multiplier.mask(text.str.contains(r"\bhectare\b", regex=True, na=False), 107639.0)

    return value * multiplier


def _parse_floor_level(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower().str.strip()
    value = _extract_first_number(text)
    is_ground = text.str.contains(r"\bground\b", regex=True, na=False)
    return value.mask(is_ground, 0.0)


def _parse_count_like(series: pd.Series) -> pd.Series:
    return _extract_first_number(series)


def _ensure_dataframe(X, columns: List[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, pd.Series):
        df = X.to_frame()
    else:
        df = pd.DataFrame(X)

    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]

    return df


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert selected text columns to numeric and drop noisy high-cardinality text features."""

    def __init__(
        self,
        drop_columns: List[str] | None = None,
        numeric_from_text: Dict[str, Callable[[pd.Series], pd.Series]] | None = None,
    ):
        self.drop_columns = drop_columns or []
        self.numeric_from_text = numeric_from_text or {}

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        self.input_columns_ = list(X_df.columns)
        self.drop_columns_ = [col for col in self.drop_columns if col in X_df.columns]
        self.numeric_from_text_ = {
            col: parser for col, parser in self.numeric_from_text.items() if col in X_df.columns
        }
        return self

    def transform(self, X):
        X_df = _ensure_dataframe(X, columns=getattr(self, "input_columns_", None))
        X_out = X_df.copy()

        for col, parser in getattr(self, "numeric_from_text_", {}).items():
            X_out[col] = parser(X_out[col])

        if getattr(self, "drop_columns_", None):
            X_out = X_out.drop(columns=self.drop_columns_, errors="ignore")

        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "input_columns_", [])
        features = [f for f in input_features if f not in set(getattr(self, "drop_columns_", []))]
        return np.asarray(features, dtype=object)


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[Pipeline, List[str], List[str]]:
    """Build preprocessing pipeline for mixed numeric/categorical features."""
    non_empty_features = X.columns[X.notna().any()].tolist()
    dropped_all_missing = [col for col in X.columns if col not in non_empty_features]

    if dropped_all_missing:
        dropped_preview = ", ".join(dropped_all_missing[:10])
        print(
            f"[WARN] Dropping {len(dropped_all_missing)} all-missing feature(s): {dropped_preview}"
        )

    if not non_empty_features:
        raise ValueError("All feature columns are fully missing.")

    X_non_empty = X[non_empty_features].copy()
    numeric_from_text = {
        "Amount(in rupees)": _parse_amount_in_rupees,
        "Carpet Area": _parse_area_to_sqft,
        "Super Area": _parse_area_to_sqft,
        "Floor": _parse_floor_level,
        "Bathroom": _parse_count_like,
        "Balcony": _parse_count_like,
        "Car Parking": _parse_count_like,
    }
    drop_columns = ["Title", "Description", "Index"]

    feature_engineering = HouseFeatureEngineer(
        drop_columns=drop_columns,
        numeric_from_text=numeric_from_text,
    )
    X_engineered = feature_engineering.fit_transform(X_non_empty)

    dropped_noise_cols = [col for col in drop_columns if col in X_non_empty.columns]
    if dropped_noise_cols:
        print(
            "[INFO] Dropping noisy text/id feature(s): "
            + ", ".join(dropped_noise_cols[:10])
        )

    converted_cols = [col for col in numeric_from_text if col in X_non_empty.columns]
    if converted_cols:
        print(
            "[INFO] Casting text feature(s) to numeric: "
            + ", ".join(converted_cols[:10])
        )

    numeric_features = X_engineered.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_engineered.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No usable features found to build preprocessor.")

    columns = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("columns", columns),
        ]
    )
    return preprocessor, numeric_features, categorical_features
