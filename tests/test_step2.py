import os
import sys
import unittest

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ld_preprocessing.step2_hourly_dst_inactive import run_step2_hourly_dst_and_inactive


class TestStep2HourlyDstInactive(unittest.TestCase):
    def test_downsample_and_inactive_filter(self):
        idx = pd.date_range("2011-01-01 00:00:00", periods=8, freq="15min")
        data = {
            "A": [1, 1, 1, 1, 2, 2, 2, 2],
            "B": [0, 0, 0, 0, 0, 0, 0, 0],
        }
        df_15m = pd.DataFrame(data, index=idx, dtype=np.float32)

        config = {
            "target_definition": "hourly_kW_mean_of_4",
            "dst_policy": "drop_entire_transition_days",
            "timezone": "Europe/Lisbon",
            "inactive_filter": {
                "nonzero_rate_min": 0.5,
                "max_consecutive_zeros_hours_max": 1,
            },
        }

        hourly_active, report = run_step2_hourly_dst_and_inactive(df_15m, config)

        self.assertEqual(hourly_active.shape, (2, 1))
        self.assertIn("A", hourly_active.columns)
        self.assertNotIn("B", hourly_active.columns)
        self.assertEqual(len(report["active_clients"]), 1)
        self.assertEqual(len(report["dropped_clients"]), 1)


if __name__ == "__main__":
    unittest.main()
