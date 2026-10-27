"""Valuation discrepancy analysis to detect undervalued vs. premium vehicles."""

from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline


def calculate_market_valuation(
    df: pd.DataFrame,
    pipeline: Pipeline,
    undervalued_threshold_pct: float = -12.0,
    overvalued_threshold_pct: float = 12.0,
) -> pd.DataFrame:
    """Evaluate market vehicles against model-predicted intrinsic values.

    Args:
        df: Enriched dataset containing required feature columns and actual 'MSRP'.
        pipeline: Trained regression pipeline.
        undervalued_threshold_pct: Percentage threshold below predicted value to qualify as a bargain.
        overvalued_threshold_pct: Percentage threshold above predicted value to qualify as a brand premium.

    Returns:
        DataFrame enriched with predicted values, dollar discrepancies, percentage margins, and rating.
    """
    evaluated = df.copy()

    # Predict intrinsic valuation
    evaluated["Predicted_MSRP"] = pipeline.predict(evaluated)

    # Discrepancy metrics (Actual vs Predicted)
    evaluated["Valuation_Gap"] = evaluated["MSRP"] - evaluated["Predicted_MSRP"]
    evaluated["Valuation_Gap_Pct"] = (
        evaluated["Valuation_Gap"] / evaluated["Predicted_MSRP"]
    ) * 100.0

    def categorize_valuation(pct: float) -> str:
        if pct <= undervalued_threshold_pct:
            return "Undervalued (High Value Deal)"
        elif pct >= overvalued_threshold_pct:
            return "Overvalued (Brand / Luxury Premium)"
        else:
            return "Fair Market Value"

    evaluated["Market_Rating"] = evaluated["Valuation_Gap_Pct"].apply(categorize_valuation)

    return evaluated


def get_top_market_deals(
    evaluated_df: pd.DataFrame, n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve top undervalued (deals) and top overvalued (premium) vehicles.

    Args:
        evaluated_df: Evaluated DataFrame from calculate_market_valuation.
        n: Number of records to return for each extreme.

    Returns:
        Tuple of (top_deals_df, top_premiums_df).
    """
    display_cols = [
        "Make",
        "Model",
        "Type",
        "Origin",
        "Horsepower",
        "Combined_MPG",
        "MSRP",
        "Predicted_MSRP",
        "Valuation_Gap",
        "Valuation_Gap_Pct",
        "Market_Rating",
    ]
    cols = [c for c in display_cols if c in evaluated_df.columns]

    top_deals = (
        evaluated_df.sort_values(by="Valuation_Gap_Pct", ascending=True)
        .head(n)[cols]
        .reset_index(drop=True)
    )

    top_premiums = (
        evaluated_df.sort_values(by="Valuation_Gap_Pct", ascending=False)
        .head(n)[cols]
        .reset_index(drop=True)
    )

    return top_deals, top_premiums
