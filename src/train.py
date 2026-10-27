"""Main training and pipeline execution script for Car Market Valuation Engine."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import clean_car_data, load_raw_data, save_processed_data
from src.features import engineer_features
from src.models import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    build_pipeline,
    evaluate_regression,
    save_pipeline,
)
from src.valuation import calculate_market_valuation, get_top_market_deals


def run_pipeline(
    raw_data_path: str = "data/raw/cars_raw.csv",
    processed_data_path: str = "data/processed/cars_cleaned.csv",
    model_output_path: str = "models/car_valuation_pipeline.joblib",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Execute end-to-end data ingestion, training, evaluation, and serialization."""
    print("=" * 65)
    print("🚗 CAR MARKET VALUATION ENGINE - TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load Data
    print(f"\n[1/5] Loading raw data from '{raw_data_path}'...")
    raw_df = load_raw_data(raw_data_path)
    print(f"      Loaded {raw_df.shape[0]} records, {raw_df.shape[1]} columns.")

    # 2. Clean Data
    print("\n[2/5] Cleaning data and applying domain-informed imputations...")
    cleaned_df = clean_car_data(raw_df)
    save_processed_data(cleaned_df, processed_data_path)
    print(f"      Processed dataset saved to '{processed_data_path}' ({len(cleaned_df)} valid vehicles).")

    # 3. Feature Engineering
    print("\n[3/5] Engineering domain features (EPA Combined MPG, Power/Weight, Margins)...")
    enriched_df = engineer_features(cleaned_df)

    # 4. Train / Test Split
    print(f"\n[4/5] Splitting data ({100 - int(test_size * 100)}% train / {int(test_size * 100)}% test)...")
    feature_cols = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES
    X = enriched_df[feature_cols]
    y = enriched_df["MSRP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"      Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Model Comparison
    print(f"\n      Benchmarking candidate architectures...")
    candidates = ["ridge", "gradient_boosting", "random_forest"]
    results = {}

    for cand in candidates:
        cand_pipe = build_pipeline(model_type=cand)
        cand_pipe.fit(X_train, y_train)
        cand_preds = cand_pipe.predict(X_test)
        cand_metrics = evaluate_regression(y_test, cand_preds)
        results[cand] = {"pipeline": cand_pipe, "metrics": cand_metrics}
        print(
            f"      - {cand.replace('_', ' ').title():<20} | "
            f"R2: {cand_metrics['r2']:.4f} | "
            f"MAE: ${cand_metrics['mae']:>8,.2f} | "
            f"RMSE: ${cand_metrics['rmse']:>8,.2f} | "
            f"MAPE: {cand_metrics['mape'] * 100:>5.2f}%"
        )

    # Select Best Model
    best_candidate_name = max(results, key=lambda k: results[k]["metrics"]["r2"])
    best_pipeline = results[best_candidate_name]["pipeline"]
    best_metrics = results[best_candidate_name]["metrics"]

    print(f"\n      🏆 Selected Champion Model: {best_candidate_name.upper()}")

    # Serialize
    saved_model_path = save_pipeline(best_pipeline, model_output_path)
    print(f"\n[5/5] Champion pipeline saved to '{saved_model_path}'")

    # Valuation Discrepancy Analysis
    print("\n" + "=" * 65)
    print("🔍 MARKET VALUATION INSIGHTS & DISCREPANCY ANALYSIS")
    print("=" * 65)
    evaluated_df = calculate_market_valuation(enriched_df, best_pipeline)
    top_deals, top_premiums = get_top_market_deals(evaluated_df, n=3)

    print("\n💰 Top 3 Undervalued Vehicles (Best Value per Spec):")
    for _, row in top_deals.iterrows():
        print(
            f"   • {row['Make']} {row['Model']} ({row['Type']}) | "
            f"Actual MSRP: ${row['MSRP']:,.0f} | "
            f"Intrinsic Val: ${row['Predicted_MSRP']:,.0f} | "
            f"Discount: {abs(row['Valuation_Gap_Pct']):.1f}%"
        )

    print("\n💎 Top 3 Premium Vehicles (Highest Brand/Luxury Markups):")
    for _, row in top_premiums.iterrows():
        print(
            f"   • {row['Make']} {row['Model']} ({row['Type']}) | "
            f"Actual MSRP: ${row['MSRP']:,.0f} | "
            f"Intrinsic Val: ${row['Predicted_MSRP']:,.0f} | "
            f"Premium: +{row['Valuation_Gap_Pct']:.1f}%"
        )

    print("\nPipeline execution completed successfully.\n")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Market Valuation Engine")
    parser.add_argument("--raw-data", default="data/raw/cars_raw.csv")
    parser.add_argument("--processed-data", default="data/processed/cars_cleaned.csv")
    parser.add_argument("--model-out", default="models/car_valuation_pipeline.joblib")
    args = parser.parse_args()

    run_pipeline(
        raw_data_path=args.raw_data,
        processed_data_path=args.processed_data,
        model_output_path=args.model_out,
    )
