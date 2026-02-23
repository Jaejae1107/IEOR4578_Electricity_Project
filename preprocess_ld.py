#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

def max_consecutive_zeros_hours(series):
    arr = series.fillna(0).eq(0).to_numpy(dtype=np.int8)
    if arr.size == 0 or arr.max() == 0:
        return 0
    padded = np.r_[0, arr, 0]
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max())

def find_transition_dates(start, end, timezone):
    naive_days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    local_noon = (naive_days + pd.Timedelta(hours=12)).tz_localize(
        timezone, ambiguous="NaT", nonexistent="shift_forward"
    )
    offsets = pd.Series([d.utcoffset() for d in local_noon], index=local_noon)
    changed = offsets != offsets.shift(1)
    changed.iloc[0] = False
    return sorted([d.isoformat() for d in local_noon[changed].tz_localize(None).date])

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    decimal = config.get("decimal", ",")
    df = pd.read_csv(config["input_path"], sep=";", decimal=decimal, low_memory=False)
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    if df.index.duplicated().any():
        df = df.groupby(level=0).mean(numeric_only=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.astype(np.float32)
    start, end = config["analysis_window"]
    start = pd.to_datetime(start)
    end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df.loc[(df.index >= start) & (df.index <= end)]

    integrity = {
        "rows": int(df.shape[0]),
        "clients": int(df.shape[1]),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "non_15m_intervals": int((df.index.to_series().diff().dropna() != pd.Timedelta(minutes=15)).sum())
    }

    hourly = df.resample("h").mean().astype(np.float32)
    dst_dates = find_transition_dates(hourly.index.min(), hourly.index.max(), config.get("timezone", "Europe/Lisbon"))
    if config["dst_policy"] == "drop_entire_transition_days":
        drop_dates = pd.to_datetime(dst_dates).normalize()
        hourly = hourly.loc[~hourly.index.normalize().isin(drop_dates)]

    nonzero_rate = hourly.fillna(0).ne(0).sum(axis=0) / len(hourly)
    max_zero_runs = hourly.apply(max_consecutive_zeros_hours, axis=0)
    nz_min = config["inactive_filter"]["nonzero_rate_min"]
    max_zero_max = config["inactive_filter"]["max_consecutive_zeros_hours_max"]
    dropped_mask = (nonzero_rate < nz_min) | (max_zero_runs > max_zero_max)

    active_clients = sorted(hourly.columns[~dropped_mask].tolist())
    dropped_clients = sorted(hourly.columns[dropped_mask].tolist())
    hourly_active = hourly[active_clients].copy()

    per_day_counts = hourly_active.groupby(hourly_active.index.normalize()).size()
    inconsistent_days = [str(d.date()) for d in per_day_counts.index[per_day_counts != 24]]

    first_nonzero = {}
    for c in hourly.columns:
        nz_idx = hourly[c].fillna(0).ne(0)
        first_nonzero[c] = str(hourly.index[nz_idx.argmax()]) if nz_idx.any() else None

    metadata = {
        "target_definition": config["target_definition"],
        "downsample_rule": "mean_of_4_consecutive_15min_kW",
        "dst_policy": config["dst_policy"],
        "dst_dropped_dates": dst_dates,
        "inactive_filter_rule": config["inactive_filter"],
        "active_clients": active_clients,
        "dropped_clients": dropped_clients,
        "analysis_window": {"start": config["analysis_window"][0], "end": config["analysis_window"][1]},
        "integrity_report": integrity,
        "inconsistent_daily_steps_after_dst_drop": inconsistent_days,
        "first_nonzero_timestamp": first_nonzero
    }

    out_master = Path(config["save"]["master_wide_hourly_path"])
    out_meta = Path(config["save"]["metadata_path"])
    out_agg = Path(config["save"]["aggregate_hourly_path"])

    hourly_active.to_parquet(out_master)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    hourly_active.sum(axis=1).rename("aggregate_hourly_kW").to_frame().to_parquet(out_agg)

    print("done", hourly_active.shape, len(active_clients), len(dropped_clients))

if __name__ == "__main__":
    main()
