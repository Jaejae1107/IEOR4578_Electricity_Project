"""
Electricity Load Forecasting Dashboard
=======================================
Interactive Streamlit dashboard to compare model predictions against
actual values for 156 electricity meter clients.

Usage:
    source .venv/bin/activate
    streamlit run dashboard/dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Load Forecasting",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model_storage"

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_END  = "2013-12-31 23:00:00"
VAL_START  = "2014-01-01 00:00:00"
VAL_END    = "2014-04-30 23:00:00"
TEST_START = "2014-05-01 00:00:00"

LOOKBACK_HOURS = 672  # 4-week lookback used by AutoARIMA/AutoETS/SARIMAX
CHUNK_H = 720         # iTransformer rolling prediction horizon (~1 month)

EXOG_COLS_ALL = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month_sin", "month_cos",
]

MODEL_NAMES = ["AutoARIMA", "AutoETS", "SARIMAX", "Prophet", "iTransformer"]

MODEL_COLORS = {
    "AutoARIMA":    "#0066FF",
    "AutoETS":      "#FF8800",
    "SARIMAX":      "#00BB44",
    "Prophet":      "#EE1111",
    "iTransformer": "#AA22FF",
}


# ── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data from parquet...")
def load_data():
    """Load master parquet and derive train/val/test splits in long format."""
    # Load wide-format parquet (timestamp as index, 156 client columns)
    df_wide = pd.read_parquet(ROOT / "master_wide_hourly.parquet")
    df_wide = df_wide.reset_index()  # timestamp becomes a column
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"])

    # Convert wide -> long
    client_cols = [c for c in df_wide.columns if c.startswith("MT_")]
    df_long = df_wide.melt(
        id_vars="timestamp", value_vars=client_cols,
        var_name="unique_id", value_name="y",
    )
    df_long = df_long.rename(columns={"timestamp": "ds"})
    df_long = df_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Generate calendar features
    df_long["hour_sin"]  = np.sin(2 * np.pi * df_long["ds"].dt.hour / 24)
    df_long["hour_cos"]  = np.cos(2 * np.pi * df_long["ds"].dt.hour / 24)
    df_long["dow_sin"]   = np.sin(2 * np.pi * df_long["ds"].dt.dayofweek / 7)
    df_long["dow_cos"]   = np.cos(2 * np.pi * df_long["ds"].dt.dayofweek / 7)
    df_long["is_weekend"] = (df_long["ds"].dt.dayofweek >= 5).astype(int)
    df_long["month_sin"] = np.sin(2 * np.pi * df_long["ds"].dt.month / 12)
    df_long["month_cos"] = np.cos(2 * np.pi * df_long["ds"].dt.month / 12)

    # Splits
    train = df_long[
        (df_long["ds"] >= "2012-01-01") & (df_long["ds"] <= TRAIN_END)
    ].copy()
    val = df_long[
        (df_long["ds"] >= VAL_START) & (df_long["ds"] <= VAL_END)
    ].copy()
    test = df_long[df_long["ds"] >= TEST_START].copy()

    clients = sorted(df_long["unique_id"].unique().tolist())

    return train, val, test, clients


def apply_lookback(df, hours):
    """Keep only the last `hours` rows per client."""
    return (
        df.groupby("unique_id", group_keys=False)
        .tail(hours)
        .reset_index(drop=True)
    )


# ── Prediction functions (each cached) ──────────────────────────────────────
@st.cache_data(show_spinner="Running AutoARIMA predictions...")
def predict_autoarima(test_h):
    import joblib
    sf = joblib.load(MODEL_DIR / "autoarima_final.joblib")
    preds = sf.predict(h=test_h)
    if "unique_id" not in preds.columns:
        preds = preds.reset_index()
    return preds[["unique_id", "ds", "AutoARIMA"]]


@st.cache_data(show_spinner="Running AutoETS predictions...")
def predict_autoets(test_h):
    import joblib
    sf = joblib.load(MODEL_DIR / "autoets_final.joblib")
    preds = sf.predict(h=test_h)
    if "unique_id" not in preds.columns:
        preds = preds.reset_index()
    return preds[["unique_id", "ds", "AutoETS"]]


@st.cache_data(show_spinner="Running SARIMAX predictions...")
def predict_sarimax(test_h, _test_exog):
    import joblib
    sf = joblib.load(MODEL_DIR / "sarimax_final.joblib")
    X_test = _test_exog[["unique_id", "ds"] + EXOG_COLS_ALL]
    preds = sf.predict(h=test_h, X_df=X_test)
    if "unique_id" not in preds.columns:
        preds = preds.reset_index()
    # SARIMAX uses AutoARIMA under the hood — rename the output column
    preds = preds.rename(columns={"AutoARIMA": "SARIMAX"})
    return preds[["unique_id", "ds", "SARIMAX"]]


@st.cache_data(show_spinner="Running Prophet predictions (156 models)...")
def predict_prophet(_test_df):
    import joblib
    prophet_models = joblib.load(MODEL_DIR / "prophet_final.joblib")
    preds_list = []
    for uid, grp in _test_df.groupby("unique_id"):
        if uid not in prophet_models:
            continue
        m = prophet_models[uid]
        future = grp[["ds", "is_weekend"]].copy()
        forecast = m.predict(future)
        pred_df = pd.DataFrame({
            "unique_id": uid,
            "ds": forecast["ds"].values,
            "Prophet": forecast["yhat"].values,
        })
        preds_list.append(pred_df)
    return pd.concat(preds_list, ignore_index=True)


def _add_calendar_features(df):
    """Add calendar exogenous features to a DataFrame with a 'ds' column."""
    df["hour_sin"]   = np.sin(2 * np.pi * df["ds"].dt.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["ds"].dt.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["ds"].dt.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["ds"].dt.dayofweek / 7)
    df["is_weekend"] = (df["ds"].dt.dayofweek >= 5).astype(int)
    df["month_sin"]  = np.sin(2 * np.pi * df["ds"].dt.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["ds"].dt.month / 12)
    return df


@st.cache_data(show_spinner="Running iTransformer rolling predictions...")
def predict_itransformer(_train_val_with_exog, _test_dates):
    from neuralforecast import NeuralForecast

    nf = NeuralForecast.load(str(MODEL_DIR / "itransformer_val"))

    remaining = set(_test_dates)
    history = _train_val_with_exog.copy()
    all_preds = []

    while remaining:
        preds = nf.predict(df=history).reset_index()
        preds["ds"] = pd.to_datetime(preds["ds"])

        # Collect predictions that match remaining test dates
        matched = preds[preds["ds"].isin(remaining)]
        if len(matched) == 0:
            break  # safety: no progress possible
        all_preds.append(matched)
        remaining -= set(matched["ds"].unique())

        # Append ALL predictions to history to keep it contiguous
        pred_rows = preds[["unique_id", "ds", "iTransformer"]].rename(
            columns={"iTransformer": "y"},
        )
        pred_rows = _add_calendar_features(pred_rows.copy())
        history = (
            pd.concat([history, pred_rows], ignore_index=True)
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

    result = pd.concat(all_preds, ignore_index=True)
    return result[["unique_id", "ds", "iTransformer"]]


def get_predictions(model_name, train, val, test):
    """Dispatch prediction for a given model name."""
    test_h = test["ds"].nunique()

    if model_name == "AutoARIMA":
        return predict_autoarima(test_h)
    elif model_name == "AutoETS":
        return predict_autoets(test_h)
    elif model_name == "SARIMAX":
        return predict_sarimax(test_h, test)
    elif model_name == "Prophet":
        return predict_prophet(test)
    elif model_name == "iTransformer":
        train_val = pd.concat([train, val], ignore_index=True)
        train_val = train_val.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        test_dates = test["ds"].unique().tolist()
        return predict_itransformer(train_val, test_dates)
    return None


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y, yhat = y_true[mask], y_pred[mask]
    if len(y) == 0:
        return {"MSE": np.nan, "MAE": np.nan, "WAPE": np.nan}
    mse = float(np.mean((y - yhat) ** 2))
    mae = float(np.mean(np.abs(y - yhat)))
    denom = float(np.sum(np.abs(y)))
    wape = float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else np.nan
    return {"MSE": mse, "MAE": mae, "WAPE": wape}


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    st.title("Electricity Load Forecasting Dashboard")

    # Load data
    train, val, test, clients = load_data()

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header("Controls")

    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=MODEL_NAMES,
        default=["AutoARIMA", "iTransformer"],
    )

    selected_client = st.sidebar.selectbox("Select Client", options=clients)

    min_date = test["ds"].min().date()
    max_date = test["ds"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if not selected_models:
        st.info("Please select at least one model from the sidebar.")
        return

    # Handle incomplete date range selection (user clicked first date but not second)
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = date_range

    # ── Load predictions for selected models ─────────────────────────────
    all_preds = {}
    for model_name in selected_models:
        preds = get_predictions(model_name, train, val, test)
        if preds is not None:
            all_preds[model_name] = preds

    # ── Merge predictions with actuals ───────────────────────────────────
    merged = test[["unique_id", "ds", "y"]].copy()
    for model_name, preds in all_preds.items():
        merged = merged.merge(
            preds[["unique_id", "ds", model_name]],
            on=["unique_id", "ds"],
            how="left",
        )

    # Filter for selected client and date range
    client_df = merged[
        (merged["unique_id"] == selected_client)
        & (merged["ds"].dt.date >= start_date)
        & (merged["ds"].dt.date <= end_date)
    ].sort_values("ds")

    # ── Per-client metrics ───────────────────────────────────────────────
    st.subheader(f"Performance Metrics — {selected_client}")

    metric_cols = st.columns(len(selected_models))
    client_metrics_rows = []
    for i, model_name in enumerate(selected_models):
        if model_name in client_df.columns:
            m = compute_metrics(client_df["y"].values, client_df[model_name].values)
            client_metrics_rows.append({"Model": model_name, **m})
            with metric_cols[i]:
                st.metric(label=f"{model_name} — WAPE", value=f"{m['WAPE']:.4f}")
                st.caption(f"MSE: {m['MSE']:,.2f}  |  MAE: {m['MAE']:,.2f}")

    # ── Overall metrics ──────────────────────────────────────────────────
    st.subheader("Overall Metrics (All Clients)")

    overall_rows = []
    for model_name in selected_models:
        if model_name in merged.columns:
            m = compute_metrics(merged["y"].values, merged[model_name].values)
            overall_rows.append({"Model": model_name, **m})

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        st.dataframe(
            overall_df.style.format({"MSE": "{:,.2f}", "MAE": "{:,.2f}", "WAPE": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Line chart ───────────────────────────────────────────────────────
    st.subheader(f"Actual vs Predicted — {selected_client}")

    fig = go.Figure()

    # Actual values — thin gray so prediction colors stand out
    fig.add_trace(go.Scatter(
        x=client_df["ds"],
        y=client_df["y"],
        name="Actual",
        mode="lines",
        line=dict(color="#333333", width=1.5),
    ))

    # Model predictions — bold vivid colors on top
    for model_name in selected_models:
        if model_name in client_df.columns:
            fig.add_trace(go.Scatter(
                x=client_df["ds"],
                y=client_df[model_name],
                name=model_name,
                mode="lines",
                line=dict(color=MODEL_COLORS.get(model_name, "#888"), width=2),
            ))

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Load (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=500,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics comparison table (per-client) ────────────────────────────
    if client_metrics_rows:
        st.subheader(f"Metrics Comparison — {selected_client}")
        metrics_df = pd.DataFrame(client_metrics_rows)
        st.dataframe(
            metrics_df.style.format({"MSE": "{:,.2f}", "MAE": "{:,.2f}", "WAPE": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
