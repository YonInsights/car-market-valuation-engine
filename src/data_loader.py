"""Data loading and cleaning utilities for the Car Market Valuation Engine."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def clean_currency(series: pd.Series) -> pd.Series:
    """Strip currency signs, commas, and whitespace, converting to float.

    Args:
        series: Pandas series containing currency strings (e.g. '$36,945 ').

    Returns:
        Cleaned numeric float series.
    """
    if series.dtype == object or isinstance(series.iloc[0], str):
        return (
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
            .astype(float)
        )
    return series.astype(float)


def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load raw car dataset from CSV.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Pandas DataFrame containing raw records.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the automotive dataset.

    - Removes empty rows.
    - Standardizes text columns.
    - Cleans currency values (MSRP and Invoice).
    - Applies domain-aware imputation for missing cylinder counts
      (e.g. rotary engines or median cylinder counts by vehicle type).

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    cleaned = df.copy()

    # Drop rows that are completely NaN
    cleaned = cleaned.dropna(how="all").reset_index(drop=True)

    # Standardize string fields
    string_cols = ["Make", "Model", "Type", "Origin", "DriveTrain"]
    for col in string_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Clean currency columns
    if "MSRP" in cleaned.columns:
        cleaned["MSRP"] = clean_currency(cleaned["MSRP"])
    if "Invoice" in cleaned.columns:
        cleaned["Invoice"] = clean_currency(cleaned["Invoice"])

    # Ensure numerical datatypes
    num_cols = [
        "EngineSize",
        "Cylinders",
        "Horsepower",
        "MPG_City",
        "MPG_Highway",
        "Weight",
        "Wheelbase",
        "Length",
    ]
    for col in num_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Domain-aware cylinder imputation:
    # Mazda RX-8 features a Wankel Rotary engine (technically 0 reciprocating cylinders,
    # or treated with segment median for regression models).
    if "Cylinders" in cleaned.columns and "Type" in cleaned.columns:
        type_cylinder_medians = cleaned.groupby("Type")["Cylinders"].transform("median")
        cleaned["Cylinders"] = cleaned["Cylinders"].fillna(type_cylinder_medians)
        # Fallback to global median if any remain
        cleaned["Cylinders"] = cleaned["Cylinders"].fillna(cleaned["Cylinders"].median())

    return cleaned


def save_processed_data(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Persist processed dataset to disk.

    Args:
        df: Processed DataFrame.
        output_path: Target CSV file destination.

    Returns:
        Resolved Path to written file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest
