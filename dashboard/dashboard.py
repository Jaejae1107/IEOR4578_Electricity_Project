"""
Electricity Load Forecasting Dashboard
=======================================
Interactive Streamlit dashboard to compare model predictions against
actual values for 156 electricity meter clients.

Usage:
    source .venv/bin/activate
    streamlit run dashboard/dashboard.py
"""

import io
import tempfile
import warnings
import zipfile
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model_weights"
ARTIFACTS = ROOT / "artifacts/clustering"

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


# ── Cluster-model helpers ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_cluster_mapping():
    return pd.read_csv(ARTIFACTS / "user_cluster_mapping.csv")


@st.cache_data(show_spinner=False)
def _compute_scale_factors():
    """Returns {user_id: scale} = user_train_mean / cluster_train_mean."""
    mapping   = _load_cluster_mapping()
    df_wide   = pd.read_parquet(ROOT / "master_wide_hourly.parquet").reset_index()
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"])
    train     = df_wide[df_wide["timestamp"] <= TRAIN_END]
    mt_cols   = [c for c in train.columns if c.startswith("MT_")]
    user_means = train[mt_cols].mean()

    scales = {}
    for cid in [0, 1]:
        users   = mapping[mapping["cluster_id"] == cid]["user_id"].tolist()
        valid   = [u for u in users if u in user_means.index]
        c_mean  = user_means[valid].mean()
        for u in valid:
            scales[u] = float(user_means[u] / c_mean)
    return scales


def _load_from_zip(zip_name, member):
    import joblib
    with zipfile.ZipFile(MODEL_DIR / zip_name) as z:
        with z.open(member) as f:
            return joblib.load(io.BytesIO(f.read()))


_itransformer_tmpdirs: dict = {}

def _get_itransformer_dir(cluster_key: str) -> str:
    if cluster_key in _itransformer_tmpdirs:
        return _itransformer_tmpdirs[cluster_key]
    tmp = tempfile.mkdtemp()
    prefix = f"itransformer_final_{cluster_key}/"
    with zipfile.ZipFile(MODEL_DIR / "itransformer_models.zip") as z:
        for member in z.namelist():
            if member.startswith(prefix):
                z.extract(member, tmp)
    path = str(Path(tmp) / f"itransformer_final_{cluster_key}")
    _itransformer_tmpdirs[cluster_key] = path
    return path


# ── Prediction functions (each cached) ──────────────────────────────────────
@st.cache_data(show_spinner="Running AutoARIMA predictions...")
def predict_autoarima(test_h):
    mapping  = _load_cluster_mapping()
    scales   = _compute_scale_factors()
    outliers = mapping[mapping["cluster_id"] == -1]["user_id"].tolist()
    all_preds = []

    for cid in [0, 1]:
        sf    = _load_from_zip("AutoARIMA_models.zip", f"models/autoarima_final_cluster_{cid}.joblib")
        preds = sf.predict(h=test_h).reset_index()
        for user in mapping[mapping["cluster_id"] == cid]["user_id"]:
            udf = preds[["ds", "AutoARIMA"]].copy()
            udf["unique_id"]  = user
            udf["AutoARIMA"] *= scales.get(user, 1.0)
            all_preds.append(udf[["unique_id", "ds", "AutoARIMA"]])

    for user in outliers:
        sf    = _load_from_zip("AutoARIMA_models.zip", f"models/autoarima_final_{user}.joblib")
        preds = sf.predict(h=test_h).reset_index()
        preds["unique_id"] = user
        all_preds.append(preds[["unique_id", "ds", "AutoARIMA"]])

    return pd.concat(all_preds, ignore_index=True)


@st.cache_data(show_spinner="Running AutoETS predictions...")
def predict_autoets(test_h):
    mapping  = _load_cluster_mapping()
    scales   = _compute_scale_factors()
    outliers = mapping[mapping["cluster_id"] == -1]["user_id"].tolist()
    all_preds = []

    for cid in [0, 1]:
        sf    = _load_from_zip("AutoETS_models.zip", f"models/autoets_final_cluster_{cid}.joblib")
        preds = sf.predict(h=test_h).reset_index()
        for user in mapping[mapping["cluster_id"] == cid]["user_id"]:
            udf = preds[["ds", "AutoETS"]].copy()
            udf["unique_id"] = user
            udf["AutoETS"]  *= scales.get(user, 1.0)
            all_preds.append(udf[["unique_id", "ds", "AutoETS"]])

    for user in outliers:
        sf    = _load_from_zip("AutoETS_models.zip", f"models/autoets_final_{user}.joblib")
        preds = sf.predict(h=test_h).reset_index()
        preds["unique_id"] = user
        all_preds.append(preds[["unique_id", "ds", "AutoETS"]])

    return pd.concat(all_preds, ignore_index=True)


@st.cache_data(show_spinner="Running SARIMAX predictions...")
def predict_sarimax(test_h, _test_exog):
    mapping  = _load_cluster_mapping()
    scales   = _compute_scale_factors()
    outliers = mapping[mapping["cluster_id"] == -1]["user_id"].tolist()
    ts_df = (
        _test_exog[["ds"] + EXOG_COLS_ALL]
        .drop_duplicates("ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    all_preds = []

    for cid in [0, 1]:
        cluster_uid = f"cluster_{cid}"
        sf  = _load_from_zip("SARIMAX_models.zip", f"models/sarimax_final_{cluster_uid}.joblib")
        X   = ts_df.copy()
        X["unique_id"] = cluster_uid
        p   = sf.predict(h=test_h, X_df=X[["unique_id", "ds"] + EXOG_COLS_ALL]).reset_index(drop=True)
        col = [c for c in p.columns if c not in ("unique_id", "ds")][0]
        for user in mapping[mapping["cluster_id"] == cid]["user_id"]:
            scaled_vals = p[col].values * scales.get(user, 1.0)
            udf = pd.DataFrame({"unique_id": user, "ds": p["ds"].values, "SARIMAX": scaled_vals})
            all_preds.append(udf)

    for user in outliers:
        sf  = _load_from_zip("SARIMAX_models.zip", f"models/sarimax_final_{user}.joblib")
        X   = ts_df.copy()
        X["unique_id"] = user
        p   = sf.predict(h=test_h, X_df=X[["unique_id", "ds"] + EXOG_COLS_ALL]).reset_index(drop=True)
        col = [c for c in p.columns if c not in ("unique_id", "ds")][0]
        p["unique_id"] = user
        all_preds.append(p[["unique_id", "ds", col]].rename(columns={col: "SARIMAX"}))

    return pd.concat(all_preds, ignore_index=True)


@st.cache_data(show_spinner="Running Prophet predictions...")
def predict_prophet(_test_df):
    mapping  = _load_cluster_mapping()
    scales   = _compute_scale_factors()
    outliers = mapping[mapping["cluster_id"] == -1]["user_id"].tolist()
    future_base = (
        _test_df[["ds", "is_weekend"]]
        .drop_duplicates("ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    all_preds = []

    for cid in [0, 1]:
        m        = _load_from_zip("Prophet_models.zip", f"models/prophet_final_cluster_{cid}.joblib")
        forecast = m.predict(future_base)
        for user in mapping[mapping["cluster_id"] == cid]["user_id"]:
            all_preds.append(pd.DataFrame({
                "unique_id": user,
                "ds":        forecast["ds"].values,
                "Prophet":   forecast["yhat"].values * scales.get(user, 1.0),
            }))

    for user in outliers:
        m        = _load_from_zip("Prophet_models.zip", f"models/prophet_final_{user}.joblib")
        forecast = m.predict(future_base)
        all_preds.append(pd.DataFrame({
            "unique_id": user,
            "ds":        forecast["ds"].values,
            "Prophet":   forecast["yhat"].values,
        }))

    return pd.concat(all_preds, ignore_index=True)


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

    mapping   = _load_cluster_mapping()
    test_set  = set(pd.to_datetime(_test_dates))
    all_preds = []

    cluster_groups = [(0, "cluster_0"), (1, "cluster_1"), (-1, "outliers")]
    for cid, cluster_key in cluster_groups:
        nf      = NeuralForecast.load(_get_itransformer_dir(cluster_key))
        users   = mapping[mapping["cluster_id"] == cid]["user_id"].tolist()
        history = (
            _train_val_with_exog[_train_val_with_exog["unique_id"].isin(users)]
            .copy()
        )
        remaining = set(test_set)
        cluster_preds = []

        while remaining:
            preds = nf.predict(df=history).reset_index()
            preds["ds"] = pd.to_datetime(preds["ds"])
            matched = preds[preds["ds"].isin(remaining)]
            if len(matched) == 0:
                break
            cluster_preds.append(matched)
            remaining -= set(matched["ds"].unique())
            pred_rows = preds[["unique_id", "ds", "iTransformer"]].rename(
                columns={"iTransformer": "y"}
            )
            pred_rows = _add_calendar_features(pred_rows.copy())
            history = (
                pd.concat([history, pred_rows], ignore_index=True)
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True)
            )

        if cluster_preds:
            all_preds.append(pd.concat(cluster_preds, ignore_index=True))

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
        return {"MAPE": np.nan, "WAPE": np.nan}
    nonzero = y != 0
    mape = float(np.mean(np.abs((y[nonzero] - yhat[nonzero]) / y[nonzero])) * 100) if nonzero.any() else np.nan
    denom = float(np.sum(np.abs(y)))
    wape = float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else np.nan
    return {"MAPE": mape, "WAPE": wape}


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")
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
    st.subheader(f"Performance Metrics — {selected_client} ({start_date} to {end_date})")

    metric_cols = st.columns(len(selected_models))
    client_metrics_rows = []
    for i, model_name in enumerate(selected_models):
        if model_name in client_df.columns:
            m = compute_metrics(client_df["y"].values, client_df[model_name].values)
            client_metrics_rows.append({"Model": model_name, **m})
            with metric_cols[i]:
                st.metric(label=f"{model_name} — MAPE", value=f"{m['MAPE']:.2f}%")
                st.caption(f"WAPE: {m['WAPE']:.4f}")

    # ── Metrics comparison table (per-client) ────────────────────────────
    if client_metrics_rows:
        st.subheader(f"Metrics Comparison — {selected_client} ({start_date} to {end_date})")
        metrics_df = pd.DataFrame(client_metrics_rows)
        st.dataframe(
            metrics_df.style.format({"MAPE": "{:.2f}%", "WAPE": "{:.4f}"}),
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


if __name__ == "__main__":
    main()
