"""Unit tests for Car Market Valuation Engine data, features, and model pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_car_data, clean_currency
from src.features import engineer_features
from src.models import build_pipeline, evaluate_regression
from src.valuation import calculate_market_valuation


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provide a minimal test DataFrame mimicking raw vehicle records."""
    return pd.DataFrame(
        {
            "Make": ["Toyota", "BMW", "Mazda", np.nan],
            "Model": ["Camry", "M3", "RX-8", np.nan],
            "Type": ["Sedan", "Sports", "Sports", np.nan],
            "Origin": ["Asia", "Europe", "Asia", np.nan],
            "DriveTrain": ["Front", "Rear", "Rear", np.nan],
            "MSRP": ["$20,000 ", "$45,000 ", "$25,000 ", np.nan],
            "Invoice": ["$18,500 ", "$41,000 ", "$23,000 ", np.nan],
            "EngineSize": [2.4, 3.2, 1.3, np.nan],
            "Cylinders": [4.0, 6.0, np.nan, np.nan],  # Mazda RX-8 rotary
            "Horsepower": [160.0, 333.0, 238.0, np.nan],
            "MPG_City": [24.0, 16.0, 18.0, np.nan],
            "MPG_Highway": [33.0, 24.0, 25.0, np.nan],
            "Weight": [3100.0, 3400.0, 3050.0, np.nan],
            "Wheelbase": [107.0, 107.0, 106.0, np.nan],
            "Length": [189.0, 176.0, 174.0, np.nan],
        }
    )


def test_clean_currency():
    """Verify currency stripping and float casting."""
    raw_series = pd.Series(["$12,500 ", "$100,000", " $450 "])
    cleaned = clean_currency(raw_series)
    assert cleaned.iloc[0] == 12500.0
    assert cleaned.iloc[1] == 100000.0
    assert cleaned.iloc[2] == 450.0


def test_clean_car_data(sample_raw_data):
    """Verify all-null row removal and rotary cylinder handling."""
    cleaned = clean_car_data(sample_raw_data)
    assert len(cleaned) == 3  # All-NaN row dropped
    assert cleaned["MSRP"].dtype in [np.float64, float]
    assert not cleaned["Cylinders"].isnull().any()


def test_engineer_features(sample_raw_data):
    """Verify feature derivations like Combined_MPG and Brand_Tier."""
    cleaned = clean_car_data(sample_raw_data)
    enriched = engineer_features(cleaned)

    assert "Combined_MPG" in enriched.columns
    assert "Power_to_Weight" in enriched.columns
    assert "Brand_Tier" in enriched.columns

    # EPA formula check: 0.55 * 24 + 0.45 * 33 = 13.2 + 14.85 = 28.05
    assert np.isclose(enriched.loc[0, "Combined_MPG"], 28.05)
    assert enriched.loc[0, "Brand_Tier"] == "Mainstream"
    assert enriched.loc[1, "Brand_Tier"] == "Luxury"


def test_pipeline_train_and_predict(sample_raw_data):
    """Verify that regression pipeline trains and produces valid predictions."""
    cleaned = clean_car_data(sample_raw_data)
    enriched = engineer_features(cleaned)

    feature_cols = [
        "EngineSize",
        "Cylinders",
        "Horsepower",
        "Weight",
        "Combined_MPG",
        "Wheelbase",
        "Length",
        "Power_to_Weight",
        "Vehicle_Footprint",
        "Make",
        "Type",
        "Origin",
        "DriveTrain",
        "Brand_Tier",
    ]
    X = enriched[feature_cols]
    y = enriched["MSRP"]

    pipe = build_pipeline(model_type="ridge")
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))


def test_valuation_categorization(sample_raw_data):
    """Verify valuation discrepancy flags bargain vs fair vs premium."""
    cleaned = clean_car_data(sample_raw_data)
    enriched = engineer_features(cleaned)

    pipe = build_pipeline(model_type="ridge")
    feature_cols = [
        "EngineSize",
        "Cylinders",
        "Horsepower",
        "Weight",
        "Combined_MPG",
        "Wheelbase",
        "Length",
        "Power_to_Weight",
        "Vehicle_Footprint",
        "Make",
        "Type",
        "Origin",
        "DriveTrain",
        "Brand_Tier",
    ]
    pipe.fit(enriched[feature_cols], enriched["MSRP"])

    evaluated = calculate_market_valuation(enriched, pipe)
    assert "Market_Rating" in evaluated.columns
    valid_ratings = {
        "Undervalued (High Value Deal)",
        "Overvalued (Brand / Luxury Premium)",
        "Fair Market Value",
    }
    assert set(evaluated["Market_Rating"].unique()).issubset(valid_ratings)
