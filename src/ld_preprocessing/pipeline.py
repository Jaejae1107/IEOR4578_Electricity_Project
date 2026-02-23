import json
from typing import Dict

from .step1_load_integrity import run_step1_load_and_integrity
from .step2_hourly_dst_inactive import run_step2_hourly_dst_and_inactive
from .step3_save_outputs import run_step3_save_master


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(config_path: str = "config.json") -> Dict:
    config = load_config(config_path)

    df_15m_wide, step1 = run_step1_load_and_integrity(config)
    df_hourly_wide_active, step2 = run_step2_hourly_dst_and_inactive(df_15m_wide, config)

    metadata = {
        "target_definition": config["target_definition"],
        "downsample_rule": "mean_of_4_consecutive_15min_kW",
        "dst_policy": config["dst_policy"],
        "dst_dropped_dates": step2["dst_dropped_dates"],
        "inactive_filter_rule": config["inactive_filter"],
        "active_clients": step2["active_clients"],
        "dropped_clients": step2["dropped_clients"],
        "analysis_window": {
            "start": config["analysis_window"][0],
            "end": config["analysis_window"][1],
        },
        "integrity_report": step1,
        "dst_detection_artifact_snapshot": step2["dst_detection_artifact_snapshot"],
        "inconsistent_daily_steps_after_dst_drop": step2["inconsistent_daily_steps_after_dst_drop"],
        "first_nonzero_timestamp": step2["first_nonzero_timestamp"],
    }

    run_step3_save_master(df_hourly_wide_active, config, metadata)

    return {
        "shape_15m_wide": tuple(df_15m_wide.shape),
        "shape_hourly_active_wide": tuple(df_hourly_wide_active.shape),
        "active_clients": len(step2["active_clients"]),
        "dropped_clients": len(step2["dropped_clients"]),
        "dst_dropped_dates": step2["dst_dropped_dates"],
        "inconsistent_days_after_dst_drop": len(step2["inconsistent_daily_steps_after_dst_drop"]),
    }
