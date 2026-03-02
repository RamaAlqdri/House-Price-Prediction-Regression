from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


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


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
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

    X_non_empty = X[non_empty_features]
    numeric_features = X_non_empty.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_non_empty.select_dtypes(exclude=["number"]).columns.tolist()

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

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features
