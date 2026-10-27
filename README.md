# 🚗 Car Market Valuation Engine & Price Intelligence

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-5%20passed%20%7C%20100%25-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-black.svg)](https://pep8.org/)

An enterprise-ready **Automotive Market Intelligence and Machine Learning Valuation Engine**. This project ingests physical, mechanical, and regional vehicle specifications to predict fair-market Manufacturer's Suggested Retail Price (**MSRP**), attributes price determinants, and detects market pricing anomalies—isolating **undervalued bargains** and **brand luxury premiums**.

---

## 📌 Executive Summary & Business Problem

In automotive retail, lending, and secondary acquisition, determining a vehicle's intrinsic fair market value based purely on engineering attributes is critical. Traditional pricing heuristics often fail to capture non-linear interactions between brand origin, horsepower-to-weight dynamics, and fuel efficiency.

### Core Business Questions Addressed:
1. **Quantitative Price Driver Discovery:** Measure how horsepower, vehicle footprint, fuel efficiency (EPA Combined MPG), and origin influence vehicle MSRP and dealer invoice margins.
2. **Performance-Efficiency Trade-off Analysis:** Quantify the trade-off frontier between engine power output and fuel economy across Asian, European, and American manufacturers.
3. **Machine Learning Valuation Engine:** Develop and validate an end-to-end regression pipeline capable of predicting MSRP with over $92\%$ accuracy ($R^2$).
4. **Market Arbitrage & Anomaly Detection:** Compute **Valuation Discrepancy Gaps** to identify:
   - **Undervalued Vehicles (High Value Deals):** Cars whose actual MSRP is substantially below their mechanical and feature worth.
   - **Overvalued Vehicles (Brand Premiums):** Cars commanding significant retail markups attributable to badge equity or luxury positioning.

---

## 🔬 Domain-Informed Engineering Highlights

* **Domain-Aware Rotary Engine Handling:** In the raw dataset, the **Mazda RX-8** lacked cylinder counts. Conventional automated imputation blindly fills this with the global mean ($\approx 5.8$ cylinders), ignoring mechanical reality. Because the Mazda RX-8 utilizes the **Wankel Renesis Rotary Engine** (which has 0 reciprocating pistons), domain-informed segment imputation was engineered to maintain mechanical fidelity.
* **EPA Standard Fuel Economy Metric:** Engineered a weighted composite efficiency metric:
  $$\text{Combined MPG} = 0.55 \times \text{MPG}_{\text{City}} + 0.45 \times \text{MPG}_{\text{Highway}}$$
* **Power-to-Weight Dynamics:** Calculated acceleration potential indicators ($\frac{\text{Horsepower}}{\text{Curb Weight}}$) to capture sporty dynamics without leaking invoice costs into predictive models.
* **Leakage-Free Scikit-Learn Pipelines:** Categorical features (`Make`, `Type`, `Origin`, `DriveTrain`, `Brand_Tier`) and numeric features are transformed within atomic `Pipeline` and `ColumnTransformer` boundaries to ensure zero data leakage across train/test splits.

---

## 📈 Exploratory Data Analysis & Market Findings

### 1. Market Composition & Distribution
* **Total Vehicles Analyzed:** 428 verified production vehicles across 38 global automakers.
* **Geographic Distribution:** Asia (36.9%, 158 models), USA (34.3%, 147 models), Europe (28.7%, 123 models).
* **Segment Breakdown:** Sedan (61.2%), SUV (14.0%), Sports (11.4%), Wagon (7.0%), Truck (5.6%), Hybrid (0.7%).
* **Pricing Dispersion:** Mean MSRP is **$\$32,775$** (median **$\$27,635$**), spanning from **$\$10,280$** (entry commuter) to **$\$192,465$** (ultra-high-performance sports car).

### 2. Regional Positioning & Fuel Economy Trade-offs
* **Asia:** Highest fleet fuel efficiency (average **24.7 Combined MPG**) with competitive pricing (average MSRP **$\$24,741$**), led by Toyota, Honda, and Nissan.
* **Europe:** Heavily concentrated in the luxury/performance tier (average MSRP **$\$48,350$**; average **243 HP**), commanding high brand equity and dealer margins.
* **USA:** Dominates large displacement engines (V8 trucks and SUVs) with balanced pricing (average MSRP **$\$28,265$**; average **218 HP**).

### 3. Dealership Profitability & Invoicing Spreads
* **Average Gross Markup:** Dealerships earn an average margin of **$\$2,760$** (**8.42%**) between Invoice and MSRP.
* **Segment Margin Spread:** European sports cars deliver the highest average gross profit per unit (**$\$4,250+$**), while Asian economy sedans operate on high-volume, lean-spread margins (7.8%).

---

## 📊 Machine Learning Model Benchmarks

Three regression architectures were evaluated on an 80/20 train-test split across 428 production vehicles:

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute % Error (MAPE) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** (L2 Regularized Baseline) | `0.8594` | $\$5,300.44$ | $\$6,875.61$ | $16.73\%$ | Baseline |
| **Random Forest Regressor** (Ensemble) | `0.9128` | $\$3,686.91$ | $\$5,415.03$ | $10.37\%$ | Candidate |
| **Gradient Boosting Regressor** (Sequential Trees) | **`0.9286`** | **$\$3,278.23$** | **$\$4,899.09$** | **`9.68%`** | **🏆 Champion** |

> **Champion Performance:** The **Gradient Boosting Regressor** captured **$92.86\%$** of price variance, achieving an average absolute prediction error of **$\$3,278$** across vehicle classes ranging from $\$10,000$ to $\$192,000+$.

---

## 🔍 Feature Importance Attribution

Feature attributions extracted from the champion Gradient Boosting model reveal the primary determinants of vehicle retail pricing:

| Feature Name | Feature Type | Importance Share (%) | Practical Business Interpretation |
| :--- | :--- | :---: | :--- |
| **`Horsepower`** | Numeric Engine Signal | **63.52%** | Primary valuation driver; power output dictates vehicle class and pricing tier. |
| **`Brand_Tier_Luxury`** | Categorical Feature | **6.79%** | Pure brand equity premium for recognized luxury marques (Porsche, BMW, Mercedes). |
| **`Origin_Europe`** | Geographic Categorical | **6.29%** | European manufacturing / import positioning premium. |
| **`Brand_Tier_Mainstream`** | Categorical Feature | **4.60%** | Discount factor for volume mainstream manufacturers. |
| **`Weight`** | Physical Measurement | **3.92%** | Vehicle curb weight correlates with structural footprint and material density. |
| **`EngineSize`** | Displacement (Liters) | **3.22%** | Cubic capacity displacement per motor. |
| **`Power_to_Weight`** | Engineered Dynamic | **2.76%** | Mechanical power density and acceleration responsiveness. |
| **`Combined_MPG`** | EPA Weighted Metric | **2.16%** | Fuel efficiency trade-off valuation adjustment. |
| **`Vehicle_Footprint`** | Dimensions (Wheelbase × Length) | **1.36%** | Exterior vehicle spatial envelope. |
| **`Wheelbase`** | Physical Dimension | **1.05%** | Cabin interior space and stability indicator. |

---

## 💡 Market Valuation Discrepancy Findings

Using the model's predictions as an intrinsic baseline, we calculate:
$$\text{Valuation Gap} = \text{MSRP} - \text{Predicted MSRP}$$

### Top 5 Undervalued Vehicles (Best Deals per Spec)
These vehicles offer premium specifications, horsepower, and features substantially exceeding their retail price point:

| Vehicle | Segment | Actual MSRP | Intrinsic Spec Value | Valuation Gap ($) | Discount (%) | Strategic Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Volvo S60 R 4dr** | Sedan | $\$37,560$ | $\$55,543$ | $-\$17,983$ | **-32.4%** | 300 HP AWD high-performance sport sedan priced like an entry sedan. |
| **Cadillac CTS VVT 4dr** | Sedan | $\$30,835$ | $\$45,433$ | $-\$14,598$ | **-32.1%** | 255 HP luxury platform offering exceptional equipment per dollar. |
| **Saturn Ion1 4dr** | Sedan | $\$10,995$ | $\$15,829$ | $-\$4,834$ | **-30.5%** | Ultra-affordable commuter delivering strong standard utility for price. |
| **Ford Mustang 2dr (convertible)** | Sports | $\$18,345$ | $\$24,946$ | $-\$6,601$ | **-26.5%** | High horsepower-to-cost ratio for an open-top sport vehicle. |
| **Audi A6 4.2 Quattro 4dr** | Sedan | $\$49,690$ | $\$64,509$ | $-\$14,819$ | **-23.0%** | V8 AWD executive sedan priced well below comparable European rivals. |

### Top 5 Premium / Overvalued Vehicles (Highest Markups)
These vehicles trade at significant retail markups above their raw mechanical and physical specifications:

| Vehicle | Segment | Actual MSRP | Intrinsic Spec Value | Valuation Gap ($) | Premium (%) | Strategic Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Subaru Baja** | Truck | $\$24,520$ | $\$19,056$ | $+\$5,464$ | **+28.7%** | Niche unibody utility truck commanding unique novelty markup. |
| **Saturn L300 2** | Wagon | $\$23,560$ | $\$18,435$ | $+\$5,125$ | **+27.8%** | Segment scarcity pricing on mid-size passenger wagon. |
| **Volvo C70 LPT convertible** | Sedan | $\$40,565$ | $\$32,037$ | $+\$8,528$ | **+26.6%** | Convertible luxury body-style premium over mechanical specs. |
| **Honda S2000 convertible** | Sports | $\$33,260$ | $\$27,190$ | $+\$6,070$ | **+22.3%** | High-RPM roadster enthusiast markup over basic 4-cylinder specs. |
| **GMC Sonoma Crew Cab** | Truck | $\$25,395$ | $\$20,850$ | $+\$4,545$ | **+21.8%** | Compact crew cab utility premium. |

---

## 🗂️ Project Structure

```text
car-market-valuation-engine/
├── data/
│   ├── raw/
│   │   └── cars_raw.csv                 # Original raw vehicle dataset
│   └── processed/
│       └── cars_cleaned.csv             # Cleaned, imputed, and verified dataset
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb      # EDA, margin analysis, and correlation heatmaps
│   ├── 02_valuation_model_development.ipynb    # Model benchmarking, feature importance, and discrepancy engine
│   └── legacy_project_4.ipynb                  # Archived original notebook
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── data_loader.py                   # Currency parsing, missing value handling, and persistence
│   ├── features.py                      # EPA fuel economy, power-to-weight, footprint, and brand tier engineering
│   ├── models.py                        # Preprocessing ColumnTransformer, Pipeline factories, and metrics
│   ├── valuation.py                     # Valuation discrepancy gap analysis and deal categorizer
│   └── train.py                         # CLI execution script for training, evaluation, and serialization
├── models/
│   └── car_valuation_pipeline.joblib    # Serialized champion model artifact
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                 # Comprehensive PyTest test suite (100% passing)
├── scripts/
│   ├── build_notebooks.py               # Notebook build and generation pipeline
│   └── print_summary.py                 # Summary extraction utility
├── .gitignore                           # Git configuration ignoring venv, checkpoints, and cache
├── LICENSE                              # MIT License
├── pyproject.toml                       # Python package configuration and pytest settings
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Executive project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YonInsights/Project-4.git car-market-valuation-engine
cd car-market-valuation-engine
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
```bash
pytest -v
```
Output:
```text
tests/test_pipeline.py::test_clean_currency PASSED                       [ 20%]
tests/test_pipeline.py::test_clean_car_data PASSED                       [ 40%]
tests/test_pipeline.py::test_engineer_features PASSED                    [ 60%]
tests/test_pipeline.py::test_pipeline_train_and_predict PASSED           [ 80%]
tests/test_pipeline.py::test_valuation_categorization PASSED             [100%]
============================== 5 passed in 1.77s ==============================
```

### 5. Execute End-to-End Training & Analysis Pipeline
```bash
python -m src.train
```

---

## 💻 Python API Usage (Inference Example)

```python
import joblib
import pandas as pd
from src.features import engineer_features

# 1. Load serialized champion pipeline
pipeline = joblib.load("models/car_valuation_pipeline.joblib")

# 2. Define target vehicle specs
new_vehicle = pd.DataFrame([{
    "Make": "Audi",
    "Model": "A4 3.0 Quattro",
    "Type": "Sedan",
    "Origin": "Europe",
    "DriveTrain": "All",
    "EngineSize": 3.0,
    "Cylinders": 6.0,
    "Horsepower": 220.0,
    "MPG_City": 18.0,
    "MPG_Highway": 26.0,
    "Weight": 3583.0,
    "Wheelbase": 104.0,
    "Length": 179.0,
}])

# 3. Engineer derived features and generate valuation
enriched_vehicle = engineer_features(new_vehicle)
predicted_price = pipeline.predict(enriched_vehicle)[0]

print(f"Estimated Fair Market MSRP: ${predicted_price:,.2f}")
# Output: Estimated Fair Market MSRP: $35,275.17
```

---

## 📈 Exploratory & Modeling Notebooks

| Notebook | Focus | View Online |
| :--- | :--- | :--- |
| **[`01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb)** | Data cleaning, Mazda rotary analysis, price distributions, dealer margins, and correlation heatmaps. | Pre-rendered with all charts |
| **[`02_valuation_model_development.ipynb`](notebooks/02_valuation_model_development.ipynb)** | Scikit-Learn pipelines, Ridge vs Random Forest vs Gradient Boosting, residual analysis, and market anomaly engine. | Pre-rendered with all charts |

---

## 👤 Author & Portfolio

* **Portfolio:** [Yonatan Insights](https://github.com/YonInsights)
* **GitHub Repository:** [car-market-valuation-engine](https://github.com/YonInsights/Project-4)
* **Contact:** Open an issue or connect on LinkedIn.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
