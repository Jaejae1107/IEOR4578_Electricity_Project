from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class IntegrityReport:
    rows: int
    clients: int
    start: str
    end: str
    duplicate_timestamps: int
    non_15m_intervals: int


def _compute_integrity_report(df_15m_wide: pd.DataFrame) -> IntegrityReport:
    diffs = df_15m_wide.index.to_series().diff().dropna()
    return IntegrityReport(
        rows=int(df_15m_wide.shape[0]),
        clients=int(df_15m_wide.shape[1]),
        start=str(df_15m_wide.index.min()),
        end=str(df_15m_wide.index.max()),
        duplicate_timestamps=int(df_15m_wide.index.duplicated().sum()),
        non_15m_intervals=int((diffs != pd.Timedelta(minutes=15)).sum()),
    )


def run_step1_load_and_integrity(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Step 1
    Input: config dict (must include input_path, analysis_window)
    Output:
      - df_15m_wide: datetime index, sorted, numeric client columns (float32)
      - step1_report: dict with integrity summary
    """
    decimal = config.get("decimal", ",")
    df_raw = pd.read_csv(config["input_path"], sep=";", decimal=decimal, low_memory=False)
    ts_col = df_raw.columns[0]
    df_raw = df_raw.rename(columns={ts_col: "timestamp"})

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    if df_raw.index.duplicated().any():
        df_raw = df_raw.groupby(level=0).mean(numeric_only=True)

    for c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    df_15m_wide = df_raw.astype(np.float32)

    start_date, end_date = config["analysis_window"]
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_15m_wide = df_15m_wide.loc[(df_15m_wide.index >= start_ts) & (df_15m_wide.index <= end_ts)]

    report = _compute_integrity_report(df_15m_wide)
    return df_15m_wide, asdict(report)
