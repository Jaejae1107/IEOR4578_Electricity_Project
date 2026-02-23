import os
import sys
import tempfile
import unittest

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ld_preprocessing.step1_load_integrity import run_step1_load_and_integrity


class TestStep1LoadIntegrity(unittest.TestCase):
    def test_step1_basic_loading_and_window(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "sample.csv")
            rows = [
                ["2011-01-01 00:00:00", "1", "0"],
                ["2011-01-01 00:15:00", "2", "0"],
                ["2011-01-01 00:30:00", "3", "1"],
                ["2011-01-01 00:45:00", "4", "1"],
                ["2011-01-01 01:00:00", "5", "1"],
            ]
            df = pd.DataFrame(rows, columns=["timestamp", "C1", "C2"])
            df.to_csv(input_path, sep=";", index=False)

            config = {
                "input_path": input_path,
                "analysis_window": ["2011-01-01", "2011-01-01"],
            }
            loaded, report = run_step1_load_and_integrity(config)

            self.assertEqual(loaded.shape, (5, 2))
            self.assertEqual(report["clients"], 2)
            self.assertEqual(report["duplicate_timestamps"], 0)
            self.assertEqual(report["non_15m_intervals"], 0)


if __name__ == "__main__":
    unittest.main()
