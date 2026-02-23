from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _find_transition_dates(start: pd.Timestamp, end: pd.Timestamp, timezone: str) -> List[str]:
    naive_days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    local_noon = (naive_days + pd.Timedelta(hours=12)).tz_localize(
        timezone, ambiguous="NaT", nonexistent="shift_forward"
    )
    offsets = pd.Series([d.utcoffset() for d in local_noon], index=local_noon)
    changed = offsets != offsets.shift(1)
    changed.iloc[0] = False
    return sorted([d.isoformat() for d in local_noon[changed].tz_localize(None).date])


def _dst_artifact_snapshot(df_15m_wide: pd.DataFrame, dst_dates: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for d in dst_dates:
        day = pd.Timestamp(d)
        seg = df_15m_wide.loc[
            (df_15m_wide.index >= day + pd.Timedelta(hours=1))
            & (df_15m_wide.index < day + pd.Timedelta(hours=2))
        ]
        if seg.empty:
            out[d] = {"rows_01_to_02": 0.0, "all_client_zero_row_rate": 0.0}
            continue
        out[d] = {
            "rows_01_to_02": float(seg.shape[0]),
            "all_client_zero_row_rate": float((seg.eq(0).all(axis=1)).mean()),
        }
    return out


def _max_consecutive_zeros_hours(series: pd.Series) -> int:
    arr = series.fillna(0).eq(0).to_numpy(dtype=np.int8)
    if arr.size == 0 or arr.max() == 0:
        return 0
    padded = np.r_[0, arr, 0]
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max())


def run_step2_hourly_dst_and_inactive(df_15m_wide: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Step 2
    Input: 15-min wide dataframe + config
    Output:
      - df_hourly_wide_active: hourly wide dataframe after DST-day drop + inactive filtering
      - step2_report: dict with dropped DST dates, active/dropped clients, and diagnostics
    """
    if config["target_definition"] != "hourly_kW_mean_of_4":
        raise ValueError("This pipeline currently supports only target_definition='hourly_kW_mean_of_4'.")

    hourly = df_15m_wide.resample("h").mean().astype(np.float32)

    timezone = config.get("timezone", "Europe/Lisbon")
    dst_dates = _find_transition_dates(hourly.index.min(), hourly.index.max(), timezone)
    dst_snapshot = _dst_artifact_snapshot(df_15m_wide, dst_dates)

    if config["dst_policy"] == "drop_entire_transition_days":
        hourly = hourly.loc[~hourly.index.normalize().isin(pd.to_datetime(dst_dates).normalize())]
    else:
        raise ValueError("Unsupported dst_policy. Expected 'drop_entire_transition_days'.")

    nonzero_rate = hourly.fillna(0).ne(0).sum(axis=0) / len(hourly)
    max_zero_runs = hourly.apply(_max_consecutive_zeros_hours, axis=0)

    nz_min = config["inactive_filter"]["nonzero_rate_min"]
    max_zeros_max = config["inactive_filter"]["max_consecutive_zeros_hours_max"]
    dropped_mask = (nonzero_rate < nz_min) | (max_zero_runs > max_zeros_max)

    active_clients = sorted(hourly.columns[~dropped_mask].tolist())
    dropped_clients = sorted(hourly.columns[dropped_mask].tolist())
    df_hourly_wide_active = hourly[active_clients].copy()

    per_day_count = df_hourly_wide_active.groupby(df_hourly_wide_active.index.normalize()).size()
    inconsistent_days = [str(d.date()) for d in per_day_count.index[per_day_count != 24]]

    first_nonzero = {}
    for c in hourly.columns:
        nz = hourly[c].fillna(0).ne(0)
        first_nonzero[c] = str(hourly.index[nz.argmax()]) if nz.any() else None

    step2_report = {
        "dst_dropped_dates": dst_dates,
        "dst_detection_artifact_snapshot": dst_snapshot,
        "active_clients": active_clients,
        "dropped_clients": dropped_clients,
        "inactive_stats": {
            "nonzero_rate": {k: float(v) for k, v in nonzero_rate.to_dict().items()},
            "max_consecutive_zeros_hours": {k: int(v) for k, v in max_zero_runs.to_dict().items()},
        },
        "inconsistent_daily_steps_after_dst_drop": inconsistent_days,
        "first_nonzero_timestamp": first_nonzero,
    }
    return df_hourly_wide_active, step2_report
