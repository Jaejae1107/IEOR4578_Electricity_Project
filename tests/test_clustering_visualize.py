import os
import sys
import tempfile
import unittest

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from clustering_pipeline.visualize import generate_clustering_visualizations


class TestClusteringVisualizations(unittest.TestCase):
    def test_generate_visualization_files(self):
        with tempfile.TemporaryDirectory() as td:
            feature_table = pd.DataFrame(
                {
                    "mean_load": [100.0, 120.0, 90.0],
                    "std_load": [20.0, 25.0, 18.0],
                    "coeff_var": [0.20, 0.21, 0.20],
                    "load_factor": [0.50, 0.55, 0.48],
                    "zero_rate": [0.0, 0.0, 0.05],
                    "weekend_ratio": [1.0, 1.1, 0.9],
                    "day_night_ratio": [1.4, 1.6, 1.0],
                    "peak_hour": [16, 17, 10],
                    **{f"hour_profile_{hour:02d}": [1.0 + hour * 0.01, 0.9 + hour * 0.01, 0.8 + hour * 0.01] for hour in range(24)},
                    **{f"weekday_profile_{weekday}": [1.0, 1.1, 0.9] for weekday in range(7)},
                    **{f"month_profile_{month:02d}": [1.0, 1.1, 0.9] for month in range(1, 13)},
                },
                index=["U1", "U2", "U3"],
            )
            mapping_df = pd.DataFrame(
                {
                    "user_id": ["U1", "U2", "U3"],
                    "cluster_id": [0, 0, -1],
                    "is_outlier": [False, False, True],
                    "cluster_name": ["regular_cluster", "regular_cluster", "outlier_irregular_profile"],
                }
            )
            summary_df = pd.DataFrame(
                {
                    "cluster_id": [-1, 0],
                    "n_users": [1, 2],
                    "cluster_mean_load": [90.0, 110.0],
                    "cluster_std_load": [18.0, 22.5],
                    "cluster_coeff_var": [0.20, 0.205],
                    "cluster_load_factor": [0.48, 0.525],
                    "cluster_zero_rate": [0.05, 0.0],
                    "cluster_weekend_ratio": [0.9, 1.05],
                    "cluster_day_night_ratio": [1.0, 1.5],
                    "cluster_peak_hour": [10.0, 16.5],
                    "n_outliers": [1, 0],
                    "cluster_name": ["outlier_irregular_profile", "regular_cluster"],
                    "cluster_interpretation": ["Outlier cluster.", "Regular daytime cluster."],
                }
            )

            feature_path = os.path.join(td, "cluster_feature_table.csv")
            mapping_path = os.path.join(td, "user_cluster_mapping.csv")
            summary_path = os.path.join(td, "cluster_summary.csv")
            output_dir = os.path.join(td, "figures")

            feature_table.to_csv(feature_path)
            mapping_df.to_csv(mapping_path, index=False)
            summary_df.to_csv(summary_path, index=False)

            generate_clustering_visualizations(
                feature_table_path=feature_path,
                mapping_path=mapping_path,
                summary_path=summary_path,
                output_dir=output_dir,
            )

            self.assertTrue(os.path.exists(os.path.join(output_dir, "cluster_size_bar.svg")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "cluster_daily_profile.svg")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "cluster_pca_scatter.svg")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "index.html")))


if __name__ == "__main__":
    unittest.main()
