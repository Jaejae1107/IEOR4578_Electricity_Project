# Electricity Load Forecasting Dashboard

A two-page Streamlit application for electricity load forecasting with 156 meter clients.

- **Page 1 — AI Forecasting Agent** (`app.py`): Natural-language chat interface powered by Google Gemini. Ask for a forecast by consumer ID and the agent automatically selects the best cluster model, runs the forecast, and renders an interactive chart.
- **Page 2 — Model Comparison** (`pages/1_Model_Comparison.py`): Interactive dashboard to compare all 5 model predictions against actual values for any client and date range.

## Model Weights

The dashboard loads pre-trained model weights from a `model_weights/` directory in the **project root**. This directory is not tracked by git and must be populated manually before running the app.

Create the directory and place the following zip files inside it:

```
<project_root>/
└── model_weights/
    ├── AutoARIMA_models.zip
    ├── AutoETS_models.zip
    ├── SARIMAX_models.zip
    ├── Prophet_models.zip
    └── itransformer_models.zip
```

Each zip contains joblib model files for cluster_0, cluster_1, and individual outlier users. These are produced by the training notebooks in `src/modeling_step4/` through `src/modeling_step6/`.

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
streamlit run dashboard/app.py
```

The app opens at `http://localhost:8501` by default. Navigate between pages using the sidebar.

## Page 1 — AI Forecasting Agent

Enter your Gemini API key in the sidebar, then type a natural-language request such as:

> "Forecast MT_001 for the next 48 hours"

The agent will:
1. Look up which cluster the consumer belongs to
2. Select the best model for that specific consumer in the current season, based on per-user per-period MAPE evaluation on the test set (stored in `artifacts/clustering/per_user_period_mape.parquet`). The calendar month is mapped to one of 4 periods: spring (Mar–Jun), peak summer (Jul–Aug), fall (Sep–Oct), winter (Nov–Feb).
3. Run the forecast and return a summary with mean/peak load, explaining why the model was chosen and what the MAPE means in plain language
4. Render a Plotly line chart and full data table inline, with metric cards showing mean load, peak load, cluster MAPE, and the client's individual MAPE

**Sidebar options:**
- **Gemini API Key:** Required. Can also be set via the `GEMINI_API_KEY` environment variable.
- **Forecast Horizon:** Optionally fix the horizon (24 / 48 / 72 / 120 / 168 hours). Defaults to "Auto (from message)" so users can specify the horizon in plain text (e.g. "next 100 days").

## Page 2 — Model Comparison

- **Model selection:** Choose any combination of AutoARIMA, AutoETS, SARIMAX, Prophet, iTransformer
- **Client selection:** Browse all 156 active clients (MT_001 through MT_370)
- **Date range filter:** Narrow the test period (May–December 2014)
- **Per-client metric cards:** MAPE (primary) and WAPE for each selected model, computed over the selected date range
- **Metrics Comparison table:** Side-by-side MAPE and WAPE for all selected models, shown above the chart
- **Interactive line chart:** Actual vs predicted values (Plotly, hover-enabled)

## How It Works

Models are loaded from `model_weights/` (zipped joblib files for statsforecast/Prophet; NeuralForecast checkpoint directories for iTransformer). All predictions are cached for the session via `@st.cache_data`.

### Cluster-based forecasting

Users are grouped into two main clusters plus 8 outliers (from `artifacts/clustering/user_cluster_mapping.csv`):

- **cluster_0** (84 users) and **cluster_1** (64 users): one model is trained per cluster on the cluster-averaged series. At prediction time, the cluster-level forecast is scaled back to each individual user using a per-user scale factor (`user_train_mean / cluster_train_mean`).
- **Outliers** (8 users: MT_124, MT_132, MT_156, MT_158, MT_159, MT_161, MT_162, MT_163): each receives its own individually trained model; no scale factor is applied.

| Model | Library | Prediction method |
|-------|---------|-------------------|
| AutoARIMA | statsforecast | `sf.predict(h=5856)` — cluster + outlier models, scale factor applied to cluster users |
| AutoETS | statsforecast | `sf.predict(h=5856)` — cluster + outlier models, scale factor applied to cluster users |
| SARIMAX | statsforecast | `sf.predict(h=5856, X_df=...)` with 7 calendar exogenous features; scale factor applied to cluster users |
| Prophet | prophet | `m.predict(future)` with `is_weekend` regressor; cluster + outlier models, scale factor applied to cluster users |
| iTransformer | neuralforecast | Rolling 720-hour chunks starting from 2014-05-01; history seeded with full train+val data (ending 2014-04-30); 3 separate NeuralForecast instances (cluster_0, cluster_1, outliers) |

### Overall MAPE scores (test set)

| Model | Overall MAPE |
|-------|-------------|
| AutoARIMA | 15.33% |
| AutoETS | 35.96% |
| SARIMAX | 17.63% |
| Prophet | 19.57% |
| iTransformer | 18.11% |

## File Structure

```
dashboard/
├── app.py                    # Main page — AI Forecasting Agent (Gemini chat)
├── agent_tools.py            # Pure-Python tool functions called by the Gemini agent
├── dashboard.py              # Model Comparison logic (loaded by Page 2)
├── pages/
│   └── 1_Model_Comparison.py # Page 2 — thin wrapper around dashboard.py
└── requirements.txt
```

## Data

The dashboard reads `master_wide_hourly.parquet` from the project root and derives train/validation/test splits and calendar features in memory. No CSV files are needed at runtime (cluster mapping is read from `artifacts/clustering/user_cluster_mapping.csv`).

## Requirements

See [requirements.txt](requirements.txt) for the full dependency list. Key packages: `streamlit`, `plotly`, `statsforecast`, `prophet`, `neuralforecast`, `google-genai`.
