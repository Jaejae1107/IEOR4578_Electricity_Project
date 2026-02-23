import json
from pathlib import Path
from typing import Dict

import pandas as pd


def run_step3_save_master(df_hourly_wide_active: pd.DataFrame, config: Dict, metadata: Dict) -> None:
    """
    Step 3
    Input: hourly active dataframe + config + metadata
    Output: writes master_wide_hourly.parquet and master_metadata.json (+ optional aggregate parquet)
    """
    out_master = Path(config["save"]["master_wide_hourly_path"])
    out_meta = Path(config["save"]["metadata_path"])
    out_agg = Path(config["save"].get("aggregate_hourly_path", ""))

    out_master.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    if str(out_agg):
        out_agg.parent.mkdir(parents=True, exist_ok=True)

    df_hourly_wide_active.to_parquet(out_master)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if str(out_agg):
        agg = df_hourly_wide_active.sum(axis=1).rename("aggregate_hourly_kW")
        agg.to_frame().to_parquet(out_agg)
