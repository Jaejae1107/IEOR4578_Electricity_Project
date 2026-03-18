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

from clustering_pipeline.evaluation import compute_equal_size_periods
from clustering_pipeline.features import build_user_feature_table
from clustering_pipeline.pipeline import run_clustering_pipeline


class TestClusteringPipeline(unittest.TestCase):
    def test_build_user_feature_table(self):
        idx = pd.date_range("2012-01-01 00:00:00", periods=24 * 14, freq="h")
        df = pd.DataFrame(
            {
                "U1": np.tile(np.arange(1, 25), 14),
                "U2": np.tile(np.arange(24, 0, -1), 14),
                "U3": np.full(len(idx), 10.0),
            },
            index=idx,
        )
        feature_table = build_user_feature_table(df)
        self.assertEqual(feature_table.shape[0], 3)
        self.assertIn("mean_load", feature_table.columns)
        self.assertIn("hour_profile_00", feature_table.columns)
        self.assertIn("weekday_profile_0", feature_table.columns)
        self.assertIn("month_profile_01", feature_table.columns)

    def test_compute_equal_size_periods(self):
        idx = pd.date_range("2014-05-01 00:00:00", periods=16, freq="h")
        periods = compute_equal_size_periods(idx, n_periods=4)
        self.assertEqual(len(periods), 4)
        self.assertEqual(periods[0].n_hours, 4)
        self.assertEqual(periods[-1].n_hours, 4)

    def test_run_clustering_pipeline_outputs_files(self):
        with tempfile.TemporaryDirectory() as td:
            idx_train = pd.date_range("2012-01-01 00:00:00", periods=24 * 14, freq="h")
            idx_validation = pd.date_range("2014-01-01 00:00:00", periods=24 * 4, freq="h")
            idx_test = pd.date_range("2014-05-01 00:00:00", periods=24 * 8, freq="h")

            train_df = pd.DataFrame(
                {
                    "U1": np.tile(np.arange(1, 25), 14),
                    "U2": np.tile(np.arange(24, 0, -1), 14),
                    "U3": np.tile([5.0] * 24, 14),
                    "U4": np.tile([8.0] * 24, 14),
                    "U5": np.tile([3.0] * 24, 14),
                    "U6": np.tile([12.0] * 24, 14),
                },
                index=idx_train,
            )
            validation_df = pd.DataFrame({col: 1.0 for col in train_df.columns}, index=idx_validation)
            test_df = pd.DataFrame({col: 1.0 for col in train_df.columns}, index=idx_test)

            train_path = os.path.join(td, "train.csv")
            validation_path = os.path.join(td, "validation.csv")
            test_path = os.path.join(td, "test.csv")
            output_dir = os.path.join(td, "outputs")

            train_df.to_csv(train_path)
            validation_df.to_csv(validation_path)
            test_df.to_csv(test_path)

            result = run_clustering_pipeline(
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                output_dir=output_dir,
                n_periods=4,
                cluster_range=range(2, 3),
                min_cluster_size=1,
                random_state=42,
            )

            self.assertEqual(result["n_users"], 6)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "cluster_feature_table.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "cluster_model_selection.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "user_cluster_mapping.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "cluster_summary.csv")))
            protocol_path = os.path.join(output_dir, "evaluation_protocol.json")
            self.assertTrue(os.path.exists(protocol_path))

            with open(protocol_path, "r", encoding="utf-8") as f:
                protocol = json.load(f)
            self.assertEqual(protocol["main_metric"], "MAPE_0_100")
            self.assertEqual(len(protocol["equal_size_test_periods"]), 4)


if __name__ == "__main__":
    unittest.main()
