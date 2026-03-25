"""
agent_tools.py
==============
Pure Python tool functions for the AI Forecasting Agent.
No Streamlit imports. Uses functools.lru_cache for caching.
"""

import io
import math
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts/clustering"
MODEL_WEIGHTS = ROOT / "model_weights"
DATA_FILE = ROOT / "master_wide_hourly.parquet"

FORECAST_START = pd.Timestamp("2014-05-01 00:00:00")
LOOKBACK_HOURS = 672
CHUNK_H = 720  # iTransformer fixed prediction horizon (matches training config)

EXOG_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend",
    "month_sin", "month_cos",
]

# Fallback MAPE table (cluster-level overall) used when per_user_period_mape.parquet is absent
_MAPE_TABLE = {
    "cluster_0": {
        "AutoARIMA":    14.97,
        "AutoETS":      14.73,
        "SARIMAX":      17.63,
        "Prophet":      19.57,
        "iTransformer": 18.11,
    },
    "cluster_1": {
        "AutoARIMA":    13.04,
        "AutoETS":      12.95,
        "SARIMAX":      17.63,
        "Prophet":      19.57,
        "iTransformer": 18.11,
    },
    "outliers": {
        "AutoARIMA":    38.86,
        "AutoETS":      467.71,
        "SARIMAX":      17.63,
        "Prophet":      19.57,
        "iTransformer": 18.11,
    },
}

# Seasonal proxy: map any calendar month to one of the 4 test period IDs
_MONTH_TO_PERIOD = {
    1: 4,   # Jan  → winter
    2: 4,   # Feb  → winter
    3: 1,   # Mar  → spring
    4: 1,   # Apr  → spring
    5: 1,   # May  → spring/early summer
    6: 1,   # Jun  → spring/early summer
    7: 2,   # Jul  → peak summer
    8: 2,   # Aug  → peak summer
    9: 3,   # Sep  → fall
    10: 3,  # Oct  → fall
    11: 4,  # Nov  → winter
    12: 4,  # Dec  → winter
}

_PERIOD_LABELS = {
    1: "spring / early summer",
    2: "peak summer",
    3: "fall",
    4: "winter",
}

# Outlier consumer IDs
_OUTLIER_IDS = {
    "MT_124", "MT_132", "MT_156", "MT_158",
    "MT_159", "MT_161", "MT_162", "MT_163",
}

# Module-level cache for iTransformer temp dirs
_itransformer_tmpdirs: dict = {}

# Module-level store for last forecast data (full timestamps + values)
_forecast_store: dict = {}


def get_last_forecast(consumer_id: str) -> dict | None:
    """Return the full forecast dict for consumer_id, or None if not available."""
    return _forecast_store.get(consumer_id)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_per_user_period_mape() -> pd.DataFrame | None:
    """Load per_user_period_mape.parquet; return None if not yet generated."""
    path = ARTIFACTS / "per_user_period_mape.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_cluster_period_mape() -> dict | None:
    """
    Load precomputed cluster-period MAPE from cluster_period_mape.parquet.
    Returns a dict keyed by (cluster_key, model, period_id) -> mean cluster MAPE,
    or None if the file is not yet generated.
    """
    path = ARTIFACTS / "cluster_period_mape.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return {
        (row["cluster_key"], row["model"], int(row["period_id"])): round(float(row["cluster_MAPE"]), 2)
        for _, row in df.iterrows()
    }


@lru_cache(maxsize=1)
def _load_cluster_mapping() -> pd.DataFrame:
    """Load and cache user_cluster_mapping.csv."""
    return pd.read_csv(ARTIFACTS / "user_cluster_mapping.csv")


@lru_cache(maxsize=8)
def _compute_scale_factors(cluster_key: str) -> dict:
    """
    Compute per-user scale factors for disaggregating cluster-level forecasts.

    Scale factor = user_mean_train / cluster_mean_train
    Training period: up to 2013-12-31 23:00 inclusive.
    Only computed for cluster_0 and cluster_1; outliers return {}.
    """
    if cluster_key == "outliers":
        return {}

    df_wide = pd.read_parquet(DATA_FILE)
    # Filter to training period
    train_mask = df_wide.index <= pd.Timestamp("2013-12-31 23:00:00")
    df_train = df_wide.loc[train_mask]

    # Identify users in this cluster
    mapping = _load_cluster_mapping()
    cluster_id = int(cluster_key.replace("cluster_", ""))
    cluster_users = mapping.loc[mapping["cluster_id"] == cluster_id, "user_id"].tolist()

    # Only keep columns that actually exist in parquet
    cluster_users = [u for u in cluster_users if u in df_train.columns]

    user_means = df_train[cluster_users].mean()           # Series: user_id -> mean
    cluster_mean = user_means.mean()                       # scalar

    if cluster_mean == 0:
        return {u: 1.0 for u in cluster_users}

    return {u: float(user_means[u] / cluster_mean) for u in cluster_users}


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features to a DataFrame that has a 'ds' datetime column.
    Adds: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, month_sin, month_cos.
    """
    df = df.copy()
    hour = df["ds"].dt.hour
    dow  = df["ds"].dt.dayofweek
    month = df["ds"].dt.month

    df["hour_sin"]  = np.sin(2 * math.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * math.pi * hour  / 24)
    df["dow_sin"]   = np.sin(2 * math.pi * dow   / 7)
    df["dow_cos"]   = np.cos(2 * math.pi * dow   / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    df["month_sin"] = np.sin(2 * math.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * math.pi * (month - 1) / 12)
    return df


def _make_future_df(unique_id: str, horizon: int) -> pd.DataFrame:
    """
    Create a future DataFrame with unique_id, ds (starting at FORECAST_START),
    and all calendar features.
    """
    timestamps = pd.date_range(start=FORECAST_START, periods=horizon, freq="h")
    df = pd.DataFrame({"unique_id": unique_id, "ds": timestamps})
    df = _add_calendar_features(df)
    return df


def _load_joblib_from_zip(zip_name: str, member_path: str):
    """
    Open a zip archive and load a joblib object from the specified member path.
    Reads the bytes first, wraps in BytesIO, then calls joblib.load.
    """
    zip_path = MODEL_WEIGHTS / zip_name
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read(member_path)
    return joblib.load(io.BytesIO(raw))


def _get_itransformer_dir(cluster_key: str) -> str:
    """
    Extract the iTransformer final model directory for cluster_key to a temp dir
    (if not already cached) and return the path string.

    cluster_key must be one of: cluster_0, cluster_1, outliers
    """
    if cluster_key in _itransformer_tmpdirs:
        return _itransformer_tmpdirs[cluster_key]

    # Map cluster_key to folder name inside zip
    folder_map = {
        "cluster_0": "itransformer_final_cluster_0",
        "cluster_1": "itransformer_final_cluster_1",
        "outliers":  "itransformer_final_outliers",
    }
    folder_name = folder_map[cluster_key]

    zip_path = MODEL_WEIGHTS / "itransformer_models.zip"
    tmpdir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.startswith(folder_name + "/"):
                zf.extract(member, tmpdir)

    model_path = str(Path(tmpdir) / folder_name)
    _itransformer_tmpdirs[cluster_key] = model_path
    return model_path


def _lookup_model_mape(consumer_id: str, cluster_key: str, model_name: str) -> dict:
    """
    Return both cluster-level and per-user MAPE for a model in the current season.

    Cluster MAPE = precomputed mean MAPE across all users in the cluster for that period/model.
    User MAPE    = MAPE for this specific consumer in that period/model.

    Falls back to the cluster-level overall table if the parquet is absent.
    Returns dict with: cluster_mape, user_mape, period_label, source.
    """
    month        = pd.Timestamp("today").month
    period_id    = _MONTH_TO_PERIOD[month]
    period_label = _PERIOD_LABELS[period_id]

    cluster_table = _load_cluster_period_mape()
    if cluster_table is not None:
        cluster_mape = cluster_table.get((cluster_key, model_name, period_id))

        mape_df  = _load_per_user_period_mape()
        user_row = mape_df[
            (mape_df["user_id"] == consumer_id) &
            (mape_df["period_id"] == period_id) &
            (mape_df["model"] == model_name)
        ]
        user_mape = round(float(user_row.iloc[0]["MAPE"]), 2) if not user_row.empty else None

        return {
            "cluster_mape": cluster_mape,
            "user_mape":    user_mape,
            "period_label": period_label,
            "source":       "per_user_evaluation",
        }

    # Fallback: cluster-level overall table (no per-user breakdown)
    return {
        "cluster_mape": _MAPE_TABLE.get(cluster_key, {}).get(model_name),
        "user_mape":    None,
        "period_label": period_label,
        "source":       "cluster_fallback",
    }


def _get_model_key(cluster_id: int, consumer_id: str) -> str:
    """
    Return the model key string:
    - cluster_id 0  -> 'cluster_0'
    - cluster_id 1  -> 'cluster_1'
    - cluster_id -1 -> consumer_id (e.g., 'MT_124')
    """
    if cluster_id == 0:
        return "cluster_0"
    elif cluster_id == 1:
        return "cluster_1"
    else:
        return consumer_id


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def lookup_consumer_cluster(consumer_id: str) -> dict:
    """
    Look up which cluster a consumer (meter) belongs to.

    Given a consumer ID like 'MT_001', returns the cluster assignment including
    cluster_id (0, 1, or -1 for outliers), cluster_name, and whether the
    consumer is an outlier. Returns an error dict if the consumer is not found.

    Args:
        consumer_id: The meter ID string, e.g. 'MT_001', 'MT_200'.

    Returns:
        dict with keys: consumer_id, cluster_id, cluster_name, is_outlier
        or {"error": "..."} if consumer_id is not found.
    """
    mapping = _load_cluster_mapping()
    row = mapping[mapping["user_id"] == consumer_id]

    if row.empty:
        return {"error": f"Consumer '{consumer_id}' not found in cluster mapping."}

    r = row.iloc[0]
    return {
        "consumer_id": consumer_id,
        "cluster_id":  int(r["cluster_id"]),
        "cluster_name": str(r["cluster_name"]),
        "is_outlier":  bool(r["is_outlier"]),
    }


def get_available_models(cluster_id: int) -> dict:
    """
    Return the available forecasting models and their MAPE scores for a cluster.

    Given a cluster_id (0, 1, or -1 for outliers), returns the list of models
    with their validation MAPE scores and identifies the best model (lowest MAPE).

    Args:
        cluster_id: Integer cluster ID. Use 0 or 1 for regular clusters,
                    -1 for the outlier group.

    Returns:
        dict with keys: cluster_key, models (dict of model->mape), best_model, best_mape.
    """
    cluster_id = int(cluster_id)  # Gemini may pass as string
    if cluster_id == -1:
        cluster_key = "outliers"
    elif cluster_id == 0:
        cluster_key = "cluster_0"
    elif cluster_id == 1:
        cluster_key = "cluster_1"
    else:
        return {"error": f"Unknown cluster_id: {cluster_id}"}

    mapes = _MAPE_TABLE[cluster_key]
    best_model = min(mapes, key=mapes.__getitem__)
    best_mape  = mapes[best_model]

    return {
        "cluster_key": cluster_key,
        "models":      mapes,
        "best_model":  best_model,
        "best_mape":   best_mape,
    }


def get_best_model(consumer_id: str, forecast_start_date: str) -> dict:
    """
    Return the best forecasting model for a specific consumer in a given season,
    based on per-user, per-period MAPE evaluation on the test set.

    Maps forecast_start_date to one of 4 seasonal periods (spring, summer, fall,
    winter) and looks up which model had the lowest MAPE for this consumer in
    that period. Also returns all model scores so the agent can explain the choice.

    Args:
        consumer_id:         Meter ID string, e.g. 'MT_001'.
        forecast_start_date: ISO date string for the forecast start, e.g. '2026-03-24'.
                             Used to determine which seasonal period applies.

    Returns:
        dict with keys: consumer_id, period_id, period_label, best_model,
        best_mape, all_model_mapes (dict of model->mape), source.
        source is 'per_user_evaluation' when the parquet is available,
        or 'cluster_fallback' when falling back to cluster-level MAPEs.
    """
    # Determine period from date
    try:
        month = pd.Timestamp(forecast_start_date).month
    except Exception:
        month = pd.Timestamp("today").month
    period_id    = _MONTH_TO_PERIOD[month]
    period_label = _PERIOD_LABELS[period_id]

    # Try per-user parquet first
    mape_df = _load_per_user_period_mape()
    if mape_df is not None:
        user_period = mape_df[
            (mape_df["user_id"] == consumer_id) & (mape_df["period_id"] == period_id)
        ]
        if not user_period.empty:
            scores = user_period.set_index("model")["MAPE"].round(2).to_dict()
            best_model = min(scores, key=scores.__getitem__)
            return {
                "consumer_id":     consumer_id,
                "period_id":       period_id,
                "period_label":    period_label,
                "best_model":      best_model,
                "best_mape":       scores[best_model],
                "all_model_mapes": scores,
                "source":          "per_user_evaluation",
            }

    # Fallback: cluster-level MAPE table
    cluster_info = lookup_consumer_cluster(consumer_id)
    if "error" in cluster_info:
        return cluster_info
    cid = cluster_info["cluster_id"]
    cluster_key = "outliers" if cid == -1 else f"cluster_{cid}"
    scores = _MAPE_TABLE[cluster_key]
    best_model = min(scores, key=scores.__getitem__)
    return {
        "consumer_id":     consumer_id,
        "period_id":       period_id,
        "period_label":    period_label,
        "best_model":      best_model,
        "best_mape":       scores[best_model],
        "all_model_mapes": scores,
        "source":          "cluster_fallback",
    }


def run_forecast(
    consumer_id: str,
    model_name: str,
    horizon_hours: int = 24,
) -> dict:
    """
    Run an electricity load forecast for a specific consumer using the chosen model.

    Loads the appropriate pre-trained model, generates forecasts starting from
    2014-05-01, disaggregates cluster-level forecasts to individual consumer
    level where applicable, and stores the full forecast in memory for charting.

    Args:
        consumer_id:   Meter ID string, e.g. 'MT_001'.
        model_name:    One of 'AutoARIMA', 'AutoETS', 'SARIMAX', 'Prophet',
                       'iTransformer'.
        horizon_hours: Number of hours to forecast (max 8760). Default 24.
                       Note: accuracy degrades for very long horizons.

    Returns:
        Summary dict with keys: consumer_id, cluster, model, horizon_hours,
        forecast_start, forecast_end, mean_forecast_kwh, peak_forecast_kwh,
        status="success".
        Returns {"error": "..."} on failure.
    """
    # Clamp horizon
    horizon_hours = min(max(1, horizon_hours), 8760)

    # --- Step 1: resolve cluster ---
    cluster_info = lookup_consumer_cluster(consumer_id)
    if "error" in cluster_info:
        return cluster_info

    cluster_id   = cluster_info["cluster_id"]
    cluster_name = cluster_info["cluster_name"]
    cluster_key  = "outliers" if cluster_id == -1 else f"cluster_{cluster_id}"
    model_key    = _get_model_key(cluster_id, consumer_id)  # e.g. 'cluster_0' or 'MT_124'

    try:
        # ------------------------------------------------------------------
        # AutoARIMA
        # ------------------------------------------------------------------
        if model_name == "AutoARIMA":
            member = f"models/autoarima_final_{model_key}.joblib"
            sf = _load_joblib_from_zip("AutoARIMA_models.zip", member)
            pred = sf.predict(h=horizon_hours)
            # pred columns: unique_id, ds, AutoARIMA
            raw_values = pred["AutoARIMA"].values.astype(float)

            # Apply per-user scale factor for non-outlier clusters
            if cluster_key != "outliers":
                scale_factors = _compute_scale_factors(cluster_key)
                scale = scale_factors.get(consumer_id, 1.0)
                raw_values = raw_values * scale

            timestamps = pred["ds"].tolist()

        # ------------------------------------------------------------------
        # AutoETS
        # ------------------------------------------------------------------
        elif model_name == "AutoETS":
            member = f"models/autoets_final_{model_key}.joblib"
            sf = _load_joblib_from_zip("AutoETS_models.zip", member)
            pred = sf.predict(h=horizon_hours)
            raw_values = pred["AutoETS"].values.astype(float)

            if cluster_key != "outliers":
                scale_factors = _compute_scale_factors(cluster_key)
                scale = scale_factors.get(consumer_id, 1.0)
                raw_values = raw_values * scale

            timestamps = pred["ds"].tolist()

        # ------------------------------------------------------------------
        # SARIMAX
        # ------------------------------------------------------------------
        elif model_name == "SARIMAX":
            member = f"models/sarimax_final_{model_key}.joblib"
            sf = _load_joblib_from_zip("SARIMAX_models.zip", member)

            future_df = _make_future_df(model_key, horizon_hours)
            exog_df = future_df[["unique_id", "ds"] + EXOG_COLS].copy()

            pred = sf.predict(h=horizon_hours, X_df=exog_df)
            col = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
            raw_values = pred[col].values.astype(float)

            if cluster_key != "outliers":
                scale_factors = _compute_scale_factors(cluster_key)
                raw_values = raw_values * scale_factors.get(consumer_id, 1.0)

            timestamps = pred["ds"].tolist()

        # ------------------------------------------------------------------
        # Prophet
        # ------------------------------------------------------------------
        elif model_name == "Prophet":
            member = f"models/prophet_final_{model_key}.joblib"
            m = _load_joblib_from_zip("Prophet_models.zip", member)

            future_timestamps = pd.date_range(
                start=FORECAST_START, periods=horizon_hours, freq="h"
            )
            future_df = pd.DataFrame({"ds": future_timestamps})
            future_df["is_weekend"] = (future_df["ds"].dt.dayofweek >= 5).astype(float)

            forecast = m.predict(future_df)
            raw_values = forecast["yhat"].values.astype(float)

            if cluster_key != "outliers":
                scale_factors = _compute_scale_factors(cluster_key)
                raw_values = raw_values * scale_factors.get(consumer_id, 1.0)

            timestamps = forecast["ds"].tolist()

        # ------------------------------------------------------------------
        # iTransformer  (already per-user, no scaling needed)
        # ------------------------------------------------------------------
        elif model_name == "iTransformer":
            from neuralforecast import NeuralForecast  # local import to avoid hard dep at startup

            model_dir = _get_itransformer_dir(cluster_key)
            nf = NeuralForecast.load(path=model_dir)

            # Build history: all train+val data ending just before FORECAST_START
            # (matches evaluation notebooks which use train_val ending 2014-04-30)
            df_wide = pd.read_parquet(DATA_FILE)
            history_wide = df_wide[df_wide.index < FORECAST_START]

            # Identify which users belong to this cluster
            mapping = _load_cluster_mapping()
            if cluster_key == "outliers":
                cluster_users = mapping.loc[
                    mapping["cluster_id"] == -1, "user_id"
                ].tolist()
            else:
                cid = int(cluster_key.replace("cluster_", ""))
                cluster_users = mapping.loc[
                    mapping["cluster_id"] == cid, "user_id"
                ].tolist()

            cluster_users = [u for u in cluster_users if u in df_wide.columns]

            # Convert wide -> long
            history_long = (
                history_wide[cluster_users]
                .reset_index()
                .melt(id_vars="timestamp", value_vars=cluster_users,
                      var_name="unique_id", value_name="y")
                .rename(columns={"timestamp": "ds"})
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True)
            )
            history_long = _add_calendar_features(history_long)

            # Rolling prediction for horizons longer than one chunk
            n_chunks = math.ceil(horizon_hours / CHUNK_H)
            collected = []
            history_roll = history_long.copy()
            for _ in range(n_chunks):
                pred_df = nf.predict(df=history_roll)
                if "unique_id" not in pred_df.columns:
                    pred_df = pred_df.reset_index()
                chunk = pred_df[pred_df["unique_id"] == consumer_id].copy()
                if chunk.empty:
                    break
                collected.append(chunk)
                # Append predictions as new history for next chunk
                new_hist = pred_df[["unique_id", "ds", "iTransformer"]].rename(
                    columns={"iTransformer": "y"}
                )
                new_hist = _add_calendar_features(new_hist)
                history_roll = pd.concat([history_roll, new_hist], ignore_index=True) \
                                 .sort_values(["unique_id", "ds"]).reset_index(drop=True)

            if not collected:
                return {"error": f"iTransformer produced no output for {consumer_id}."}

            user_pred = pd.concat(collected).sort_values("ds").iloc[:horizon_hours]
            raw_values = user_pred["iTransformer"].values.astype(float)
            timestamps = user_pred["ds"].tolist()

        else:
            return {"error": f"Unknown model '{model_name}'. Choose from AutoARIMA, AutoETS, SARIMAX, Prophet, iTransformer."}

    except Exception as exc:
        return {"error": f"Forecast failed for {consumer_id} with {model_name}: {exc}"}

    # --- Post-process ---
    raw_values = np.clip(raw_values, 0, None)
    raw_values = np.round(raw_values, 3)

    # Ensure timestamps are Python-native strings for JSON serialisability
    ts_str = [
        str(t) if not isinstance(t, str) else t
        for t in timestamps
    ]

    # Look up MAPE for this consumer/cluster/model
    mape_info = _lookup_model_mape(consumer_id, cluster_key, model_name)

    # Store full forecast
    _forecast_store[consumer_id] = {
        "consumer_id":    consumer_id,
        "cluster":        cluster_name,
        "model":          model_name,
        "timestamps":     ts_str,
        "values":         raw_values.tolist(),
        "forecast_start": ts_str[0] if ts_str else str(FORECAST_START),
        "forecast_end":   ts_str[-1] if ts_str else "",
        "cluster_mape":   mape_info["cluster_mape"],
        "user_mape":      mape_info["user_mape"],
        "period_label":   mape_info["period_label"],
        "mape_source":    mape_info["source"],
    }

    mean_kwh = float(np.mean(raw_values))
    peak_kwh = float(np.max(raw_values))

    return {
        "consumer_id":      consumer_id,
        "cluster":          cluster_name,
        "model":            model_name,
        "horizon_hours":    horizon_hours,
        "forecast_start":   ts_str[0] if ts_str else str(FORECAST_START),
        "forecast_end":     ts_str[-1] if ts_str else "",
        "mean_forecast_kwh": round(mean_kwh, 3),
        "peak_forecast_kwh": round(peak_kwh, 3),
        "status":           "success",
    }
