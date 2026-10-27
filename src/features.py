"""Feature engineering transformations for automotive market data."""

import pandas as pd


LUXURY_BRANDS = {
    "Porsche",
    "Mercedes-Benz",
    "BMW",
    "Audi",
    "Lexus",
    "Jaguar",
    "Cadillac",
    "Lincoln",
    "Land Rover",
    "Acura",
    "Infiniti",
    "Saab",
    "Volvo",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive mechanical, economic, and segment features.

    Features created:
    - Combined_MPG: EPA weighted fuel economy standard (55% city / 45% highway).
    - Power_to_Weight: Ratio of horsepower to vehicle curb weight.
    - Dealer_Markup: Absolute dollar difference between MSRP and Invoice.
    - Dealer_Margin_Pct: Dealer profit margin percentage.
    - Engine_Displacement_per_Cyl: Engine size displacement per cylinder.
    - Vehicle_Footprint: Normalized dimension footprint (Wheelbase * Length / 1000).
    - Brand_Tier: Categorization into 'Luxury' or 'Mainstream'.

    Args:
        df: Cleaned car DataFrame.

    Returns:
        DataFrame enriched with engineered features.
    """
    data = df.copy()

    # EPA Combined Fuel Economy
    if "MPG_City" in data.columns and "MPG_Highway" in data.columns:
        data["Combined_MPG"] = 0.55 * data["MPG_City"] + 0.45 * data["MPG_Highway"]

    # Mechanical Performance Metrics
    if "Horsepower" in data.columns and "Weight" in data.columns:
        data["Power_to_Weight"] = data["Horsepower"] / data["Weight"].replace(0, float("nan"))

    if "EngineSize" in data.columns and "Cylinders" in data.columns:
        safe_cyl = data["Cylinders"].replace(0, float("nan"))
        data["Displacement_per_Cyl"] = data["EngineSize"] / safe_cyl

    # Vehicle Dimensions
    if "Wheelbase" in data.columns and "Length" in data.columns:
        data["Vehicle_Footprint"] = (data["Wheelbase"] * data["Length"]) / 1000.0

    # Economic & Dealership Profitability Metrics
    if "MSRP" in data.columns and "Invoice" in data.columns:
        data["Dealer_Markup"] = data["MSRP"] - data["Invoice"]
        data["Dealer_Margin_Pct"] = (data["Dealer_Markup"] / data["MSRP"].replace(0, float("nan"))) * 100.0

    # Brand Market Classification
    if "Make" in data.columns:
        data["Brand_Tier"] = data["Make"].apply(lambda make: "Luxury" if make in LUXURY_BRANDS else "Mainstream")

    return data
