from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _normalized_profile(series: pd.Series, group_index: pd.Index, scale: float) -> pd.Series:
    if scale <= 0 or pd.isna(scale):
        scale = 1.0
    return series.groupby(group_index).mean().astype(float) / scale


def build_user_feature_table(train_wide: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(train_wide.index, pd.DatetimeIndex):
        raise TypeError("train_wide index must be a DatetimeIndex.")

    feature_rows = []
    hours = list(range(24))
    weekdays = list(range(7))
    months = list(range(1, 13))

    for user_id in train_wide.columns:
        series = train_wide[user_id].astype(float)
        positive_values = series[series > 0]
        scale = float(positive_values.mean()) if not positive_values.empty else 1.0

        hour_profile = _normalized_profile(series, train_wide.index.hour, scale).reindex(hours, fill_value=0.0)
        weekday_profile = _normalized_profile(series, train_wide.index.dayofweek, scale).reindex(
            weekdays, fill_value=0.0
        )
        month_profile = _normalized_profile(series, train_wide.index.month, scale).reindex(months, fill_value=0.0)

        mean_load = float(series.mean())
        std_load = float(series.std(ddof=0))
        max_load = float(series.max())
        min_load = float(series.min())
        zero_rate = float((series == 0).mean())
        nonzero_rate = float((series != 0).mean())
        coeff_var = _safe_divide(std_load, mean_load)
        load_factor = _safe_divide(mean_load, max_load)

        weekday_mask = train_wide.index.dayofweek < 5
        weekend_mask = train_wide.index.dayofweek >= 5
        daytime_mask = (train_wide.index.hour >= 8) & (train_wide.index.hour < 20)
        nighttime_mask = ~daytime_mask

        weekday_mean = float(series[weekday_mask].mean())
        weekend_mean = float(series[weekend_mask].mean())
        daytime_mean = float(series[daytime_mask].mean())
        nighttime_mean = float(series[nighttime_mask].mean())
        peak_hour = int(hour_profile.idxmax())

        row: Dict[str, float | int | str] = {
            "user_id": user_id,
            "mean_load": mean_load,
            "std_load": std_load,
            "max_load": max_load,
            "min_load": min_load,
            "coeff_var": coeff_var,
            "load_factor": load_factor,
            "zero_rate": zero_rate,
            "nonzero_rate": nonzero_rate,
            "weekday_mean": weekday_mean,
            "weekend_mean": weekend_mean,
            "weekend_ratio": _safe_divide(weekend_mean, weekday_mean),
            "daytime_mean": daytime_mean,
            "nighttime_mean": nighttime_mean,
            "day_night_ratio": _safe_divide(daytime_mean, nighttime_mean),
            "peak_hour": peak_hour,
            "log_mean_load": float(np.log1p(max(mean_load, 0.0))),
        }

        for hour in hours:
            row[f"hour_profile_{hour:02d}"] = float(hour_profile.loc[hour])
        for weekday in weekdays:
            row[f"weekday_profile_{weekday}"] = float(weekday_profile.loc[weekday])
        for month in months:
            row[f"month_profile_{month:02d}"] = float(month_profile.loc[month])

        feature_rows.append(row)

    feature_table = pd.DataFrame(feature_rows).set_index("user_id").sort_index()
    return feature_table

