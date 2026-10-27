"""Machine learning valuation pipeline and modeling utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_NUMERIC_FEATURES = [
    "EngineSize",
    "Cylinders",
    "Horsepower",
    "Weight",
    "Combined_MPG",
    "Wheelbase",
    "Length",
    "Power_to_Weight",
    "Vehicle_Footprint",
]

DEFAULT_CATEGORICAL_FEATURES = ["Make", "Type", "Origin", "DriveTrain", "Brand_Tier"]


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Construct Scikit-learn preprocessing ColumnTransformer.

    Standardizes numeric signals and applies one-hot encoding
    to categorical dimensions with unknown-category tolerance.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Configured ColumnTransformer instance.
    """
    nums = numeric_features or DEFAULT_NUMERIC_FEATURES
    cats = categorical_features or DEFAULT_CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ]
    )
    return preprocessor


def build_pipeline(
    model_type: str = "random_forest",
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    **model_kwargs: Any,
) -> Pipeline:
    """Build an end-to-end regression pipeline.

    Args:
        model_type: One of 'random_forest', 'gradient_boosting', or 'ridge'.
        numeric_features: Numeric predictor names.
        categorical_features: Categorical predictor names.
        **model_kwargs: Hyperparameters passed to the regressor.

    Returns:
        Complete Scikit-Learn Pipeline.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model_type_lower = model_type.lower()
    if model_type_lower == "random_forest":
        kwargs = {"n_estimators": 150, "random_state": 42, "n_jobs": -1}
        kwargs.update(model_kwargs)
        regressor = RandomForestRegressor(**kwargs)
    elif model_type_lower == "gradient_boosting":
        kwargs = {"n_estimators": 120, "learning_rate": 0.08, "random_state": 42}
        kwargs.update(model_kwargs)
        regressor = GradientBoostingRegressor(**kwargs)
    elif model_type_lower == "ridge":
        kwargs = {"alpha": 1.0, "random_state": 42}
        kwargs.update(model_kwargs)
        regressor = Ridge(**kwargs)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'random_forest', 'gradient_boosting', or 'ridge'."
        )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    return pipeline


def evaluate_regression(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Compute standard regression performance benchmarks.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary containing R2, MAE, RMSE, and MAPE scores.
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def save_pipeline(pipeline: Pipeline, filepath: Union[str, Path]) -> Path:
    """Serialize trained pipeline to disk using joblib.

    Args:
        pipeline: Fitted pipeline object.
        filepath: Target destination path.

    Returns:
        Path of the saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(filepath: Union[str, Path]) -> Pipeline:
    """Load serialized pipeline artifact from disk.

    Args:
        filepath: Source artifact path.

    Returns:
        Loaded Pipeline.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)
