from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class EvaluationPeriod:
    period_id: int
    start: str
    end: str
    n_hours: int


def compute_equal_size_periods(index: pd.DatetimeIndex, n_periods: int = 4) -> List[EvaluationPeriod]:
    if n_periods not in (3, 4):
        raise ValueError("n_periods must be 3 or 4.")
    if len(index) == 0:
        raise ValueError("Cannot build evaluation periods from an empty index.")
    if len(index) % n_periods != 0:
        raise ValueError("Test set length must be divisible by n_periods for equal-size periods.")

    sorted_index = pd.DatetimeIndex(index).sort_values()
    chunk_size = len(sorted_index) // n_periods
    periods = []

    for idx in range(n_periods):
        start = sorted_index[idx * chunk_size]
        end = sorted_index[(idx + 1) * chunk_size - 1]
        periods.append(
            EvaluationPeriod(
                period_id=idx + 1,
                start=str(start),
                end=str(end),
                n_hours=chunk_size,
            )
        )

    return periods


def mape_0_100(actual: pd.Series, predicted: pd.Series) -> float:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def periods_to_records(periods: List[EvaluationPeriod]) -> List[dict]:
    return [asdict(period) for period in periods]

