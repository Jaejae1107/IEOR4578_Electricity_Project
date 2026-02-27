# Electricity Load Forecasting Dashboard

Interactive Streamlit dashboard to compare forecasting model predictions against actual values for 156 electricity meter clients.

## Setup

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r dashboard/requirements.txt
```

## Run

```bash
source .venv/bin/activate
streamlit run dashboard/dashboard.py
```

The app opens at `http://localhost:8501` by default.

## Features

- **Model selection:** Choose any combination of AutoARIMA, AutoETS, SARIMAX, Prophet, and iTransformer
- **Client selection:** Browse all 156 active clients (MT_001 through MT_370)
- **Date range filter:** Narrow the test period (May–December 2014)
- **Per-client metrics:** MSE, MAE, WAPE for the selected client
- **Overall metrics:** MSE, MAE, WAPE aggregated across all 156 clients
- **Interactive line chart:** Actual vs predicted values (Plotly, hover-enabled)

## How It Works

Models are loaded from `model_storage/` and predictions are generated on-demand when selected. Each prediction is cached for the session so toggling models on/off is instant after the first run.

| Model | Library | Prediction Method |
|-------|---------|-------------------|
| AutoARIMA | statsforecast | `sf.predict(h=5856)` |
| AutoETS | statsforecast | `sf.predict(h=5856)` |
| SARIMAX | statsforecast | `sf.predict(h=5856, X_df=...)` with calendar exogenous features |
| Prophet | prophet | Per-client `m.predict(future)` (156 independent models) |
| iTransformer | neuralforecast | Rolling 720-hour chunks appended to history |

## Data

The dashboard reads `master_wide_hourly.parquet` from the project root and derives train/validation/test splits and calendar features in memory. No CSV files are needed.

## Requirements

See [requirements.txt](requirements.txt) for the full dependency list.
