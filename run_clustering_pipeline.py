#!/usr/bin/env python3
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from clustering_pipeline.pipeline import run_clustering_pipeline
from clustering_pipeline.visualize import generate_clustering_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering pipeline on train-only electricity data.")
    parser.add_argument(
        "--train",
        default="master_wide_hourly_train_2012_2013.csv",
        help="Path to train wide CSV",
    )
    parser.add_argument(
        "--validation",
        default="master_wide_hourly_validation_2014_01_04.csv",
        help="Path to validation wide CSV",
    )
    parser.add_argument(
        "--test",
        default="master_wide_hourly_test_2014_05_12.csv",
        help="Path to test wide CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/clustering",
        help="Directory to store clustering outputs",
    )
    parser.add_argument(
        "--n-periods",
        default=4,
        type=int,
        choices=[3, 4],
        help="Number of equal-size test periods",
    )
    args = parser.parse_args()

    result = run_clustering_pipeline(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_dir=args.output_dir,
        n_periods=args.n_periods,
    )
    generate_clustering_visualizations(
        feature_table_path=os.path.join(args.output_dir, "cluster_feature_table.csv"),
        mapping_path=os.path.join(args.output_dir, "user_cluster_mapping.csv"),
        summary_path=os.path.join(args.output_dir, "cluster_summary.csv"),
        output_dir=os.path.join(args.output_dir, "figures"),
    )

    print("=== Clustering Pipeline Completed ===")
    print(f"Users clustered: {result['n_users']}")
    print(f"Outliers separated before clustering: {result['n_outliers']}")
    print(f"Selected number of clusters: {result['best_k']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Figure index: {os.path.join(args.output_dir, 'figures', 'index.html')}")
    for period in result["test_periods"]:
        print(
            f"Period {period['period_id']}: {period['start']} -> {period['end']} "
            f"({period['n_hours']} hours)"
        )


if __name__ == "__main__":
    main()
