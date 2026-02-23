import json
import os
import sys
import tempfile
import unittest

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ld_preprocessing.pipeline import run_pipeline


class TestPipeline(unittest.TestCase):
    def test_pipeline_end_to_end_on_tiny_data(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "tiny.csv")
            out_master = os.path.join(td, "master.parquet")
            out_meta = os.path.join(td, "meta.json")
            out_agg = os.path.join(td, "agg.parquet")

            idx = pd.date_range("2011-01-01 00:00:00", periods=8, freq="15min")
            df = pd.DataFrame({
                "timestamp": idx,
                "A": [1, 1, 1, 1, 2, 2, 2, 2],
                "B": [0, 0, 0, 0, 0, 0, 0, 0],
            })
            df.to_csv(input_path, sep=";", index=False)

            config_path = os.path.join(td, "config.json")
            config = {
                "input_path": input_path,
                "target_definition": "hourly_kW_mean_of_4",
                "analysis_window": ["2011-01-01", "2011-01-01"],
                "timezone": "Europe/Lisbon",
                "dst_policy": "drop_entire_transition_days",
                "inactive_filter": {
                    "nonzero_rate_min": 0.5,
                    "max_consecutive_zeros_hours_max": 1,
                },
                "save": {
                    "master_wide_hourly_path": out_master,
                    "metadata_path": out_meta,
                    "aggregate_hourly_path": out_agg,
                },
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)

            result = run_pipeline(config_path)

            self.assertEqual(result["shape_15m_wide"], (8, 2))
            self.assertEqual(result["shape_hourly_active_wide"], (2, 1))
            self.assertTrue(os.path.exists(out_master))
            self.assertTrue(os.path.exists(out_meta))
            self.assertTrue(os.path.exists(out_agg))


if __name__ == "__main__":
    unittest.main()
