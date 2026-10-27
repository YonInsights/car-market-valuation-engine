"""Print detailed model metrics, feature importances, and market anomalies."""

import joblib
import pandas as pd
from src.data_loader import clean_car_data, load_raw_data
from src.features import engineer_features
from src.models import DEFAULT_CATEGORICAL_FEATURES, DEFAULT_NUMERIC_FEATURES
from src.valuation import calculate_market_valuation, get_top_market_deals

pipe = joblib.load("models/car_valuation_pipeline.joblib")
reg = pipe.named_steps["regressor"]
prep = pipe.named_steps["preprocessor"]
cats = list(prep.named_transformers_["cat"].get_feature_names_out(DEFAULT_CATEGORICAL_FEATURES))
all_features = DEFAULT_NUMERIC_FEATURES + cats

df_imp = pd.Series(reg.feature_importances_, index=all_features).sort_values(ascending=False)
print("--- TOP 10 FEATURE IMPORTANCES ---")
for feat, imp in df_imp.head(10).items():
    print(f"| `{feat}` | {imp * 100:.2f}% |")

df = load_raw_data("data/raw/cars_raw.csv")
df = clean_car_data(df)
df = engineer_features(df)
val_df = calculate_market_valuation(df, pipe)
top_deals, top_premiums = get_top_market_deals(val_df, n=5)

print("\n--- TOP 5 BARGAINS (UNDERVALUED) ---")
for _, r in top_deals.iterrows():
    print(
        f"| **{r['Make']} {r['Model']}** | {r['Type']} | ${r['MSRP']:,.0f} | ${r['Predicted_MSRP']:,.0f} | ${r['Valuation_Gap']:,.0f} | {r['Valuation_Gap_Pct']:.1f}% |"
    )

print("\n--- TOP 5 PREMIUMS (OVERVALUED) ---")
for _, r in top_premiums.iterrows():
    print(
        f"| **{r['Make']} {r['Model']}** | {r['Type']} | ${r['MSRP']:,.0f} | ${r['Predicted_MSRP']:,.0f} | +${r['Valuation_Gap']:,.0f} | +{r['Valuation_Gap_Pct']:.1f}% |"
    )
