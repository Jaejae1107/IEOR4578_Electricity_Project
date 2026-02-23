#!/usr/bin/env python3
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ld_preprocessing.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LD2011-2014 preprocessing Steps 1-3.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    result = run_pipeline(args.config)
    print("=== Preprocessing Completed (Step 1-3) ===")
    print(f"15m wide shape: {result['shape_15m_wide']}")
    print(f"Hourly active wide shape: {result['shape_hourly_active_wide']}")
    print(f"Active clients: {result['active_clients']}")
    print(f"Dropped clients: {result['dropped_clients']}")
    print(f"DST dropped dates: {result['dst_dropped_dates']}")
    if result["inconsistent_days_after_dst_drop"] == 0:
        print("All retained days have 24 hourly steps.")
    else:
        print(f"Inconsistent daily step days: {result['inconsistent_days_after_dst_drop']}")


if __name__ == "__main__":
    main()
