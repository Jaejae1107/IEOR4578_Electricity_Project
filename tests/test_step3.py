import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ld_preprocessing.step3_save_outputs import run_step3_save_master


class TestStep3SaveOutputs(unittest.TestCase):
    def test_output_files_are_written(self):
        with tempfile.TemporaryDirectory() as td:
            idx = pd.date_range("2011-01-01 00:00:00", periods=2, freq="h")
            df = pd.DataFrame({"A": [1.0, 2.0], "B": [0.5, 0.7]}, index=idx, dtype=np.float32)

            config = {
                "save": {
                    "master_wide_hourly_path": os.path.join(td, "master.parquet"),
                    "metadata_path": os.path.join(td, "meta.json"),
                    "aggregate_hourly_path": os.path.join(td, "agg.parquet"),
                }
            }
            metadata = {"key": "value"}

            run_step3_save_master(df, config, metadata)

            self.assertTrue(os.path.exists(config["save"]["master_wide_hourly_path"]))
            self.assertTrue(os.path.exists(config["save"]["metadata_path"]))
            self.assertTrue(os.path.exists(config["save"]["aggregate_hourly_path"]))

            with open(config["save"]["metadata_path"], "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            self.assertEqual(loaded_meta["key"], "value")


if __name__ == "__main__":
    unittest.main()
