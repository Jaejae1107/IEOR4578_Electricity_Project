from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from .evaluation import compute_equal_size_periods, periods_to_records
from .features import build_user_feature_table


BASE_CLUSTER_FEATURES = [
    "log_mean_load",
    "coeff_var",
    "load_factor",
    "zero_rate",
    "weekend_ratio",
    "day_night_ratio",
]


def load_wide_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return df


def _feature_columns(feature_table: pd.DataFrame) -> list[str]:
    profile_cols = [
        col
        for col in feature_table.columns
        if col.startswith("hour_profile_")
        or col.startswith("weekday_profile_")
        or col.startswith("month_profile_")
    ]
    return sorted(profile_cols) + ["weekend_ratio", "day_night_ratio", "peak_hour"]


def _outlier_columns(feature_table: pd.DataFrame) -> list[str]:
    return _feature_columns(feature_table) + BASE_CLUSTER_FEATURES


def _cluster_name(row: pd.Series, load_quantiles: Tuple[float, float]) -> str:
    if row["cluster_id"] == -1:
        return "outlier_irregular_profile"

    low_q, high_q = load_quantiles
    if row["cluster_mean_load"] <= low_q:
        load_label = "low"
    elif row["cluster_mean_load"] >= high_q:
        load_label = "high"
    else:
        load_label = "mid"

    if row["cluster_day_night_ratio"] >= 1.15:
        shape_label = "daytime"
    elif row["cluster_day_night_ratio"] <= 0.90:
        shape_label = "nighttime"
    else:
        shape_label = "balanced"

    if row["cluster_weekend_ratio"] >= 1.05:
        weekend_label = "weekend_heavy"
    elif row["cluster_weekend_ratio"] <= 0.95:
        weekend_label = "weekday_heavy"
    else:
        weekend_label = "mixed_week"

    return f"{load_label}_load_{shape_label}_{weekend_label}"


def select_cluster_count(
    feature_table: pd.DataFrame,
    cluster_counts: Iterable[int],
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame, np.ndarray]:
    used_columns = _feature_columns(feature_table)
    X = feature_table[used_columns].fillna(0.0).to_numpy()
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    selection_rows = []
    best_k = None
    best_score = float("-inf")
    best_labels = None

    for k in cluster_counts:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_scaled)
        counts = pd.Series(labels).value_counts().sort_index()
        min_size = int(counts.min())
        score = float(silhouette_score(X_scaled, labels))
        valid = min_size >= min_cluster_size
        balance_ratio = float(min_size / counts.max())

        selection_rows.append(
            {
                "n_clusters": k,
                "silhouette_score": score,
                "min_cluster_size": min_size,
                "max_cluster_size": int(counts.max()),
                "balance_ratio": balance_ratio,
                "is_valid_under_min_cluster_size_rule": valid,
            }
        )

        combined_score = score + (balance_ratio * 0.15)

        if valid and combined_score > best_score:
            best_score = combined_score
            best_k = k
            best_labels = labels

    if best_k is None:
        best_row = max(selection_rows, key=lambda row: row["silhouette_score"] + (row["balance_ratio"] * 0.15))
        best_k = int(best_row["n_clusters"])
        model = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
        best_labels = model.fit_predict(X_scaled)

    return best_k, pd.DataFrame(selection_rows), best_labels


def detect_outliers(
    feature_table: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    used_columns = _outlier_columns(feature_table)
    X = feature_table[used_columns].fillna(0.0).to_numpy()
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    detector = IsolationForest(contamination=contamination, random_state=random_state)
    labels = detector.fit_predict(X_scaled)
    return pd.Series(labels == -1, index=feature_table.index, name="is_outlier")


def summarize_clusters(feature_table: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = feature_table.reset_index() if "user_id" not in feature_table.columns else feature_table.copy()
    merged = mapping_df.merge(feature_frame, on="user_id", how="left")
    summary = (
        merged.groupby("cluster_id")
        .agg(
            n_users=("user_id", "count"),
            cluster_mean_load=("mean_load", "mean"),
            cluster_std_load=("std_load", "mean"),
            cluster_coeff_var=("coeff_var", "mean"),
            cluster_load_factor=("load_factor", "mean"),
            cluster_zero_rate=("zero_rate", "mean"),
            cluster_weekend_ratio=("weekend_ratio", "mean"),
            cluster_day_night_ratio=("day_night_ratio", "mean"),
            cluster_peak_hour=("peak_hour", "mean"),
            n_outliers=("is_outlier", "sum"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )

    load_quantiles = (
        float(summary["cluster_mean_load"].quantile(1 / 3)),
        float(summary["cluster_mean_load"].quantile(2 / 3)),
    )
    summary["cluster_name"] = summary.apply(_cluster_name, axis=1, load_quantiles=load_quantiles)
    summary["cluster_interpretation"] = summary.apply(
        lambda row: (
            "Users in this group were separated as outliers because their training-period profiles "
            "are materially different from the main population."
            if row["cluster_id"] == -1
            else (
                f"Average load is {row['cluster_name'].split('_')[0]}, "
                f"usage is {row['cluster_name'].split('_')[2]}-leaning, "
                f"and the weekly pattern is {row['cluster_name'].split('_', 3)[3]}."
            )
        ),
        axis=1,
    )
    return summary


def run_clustering_pipeline(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    n_periods: int = 4,
    cluster_range: Iterable[int] = range(2, 9),
    min_cluster_size: int = 5,
    outlier_contamination: float = 0.05,
    random_state: int = 42,
) -> Dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = load_wide_csv(train_path)
    validation_df = load_wide_csv(validation_path)
    test_df = load_wide_csv(test_path)

    feature_table = build_user_feature_table(train_df)
    outlier_flags = detect_outliers(
        feature_table=feature_table,
        contamination=outlier_contamination,
        random_state=random_state,
    )
    regular_features = feature_table.loc[~outlier_flags].copy()
    effective_cluster_range = [k for k in cluster_range if 2 <= k < len(regular_features)]
    if not effective_cluster_range:
        raise ValueError("Not enough regular users remain after outlier filtering to fit clustering.")

    best_k, selection_df, labels = select_cluster_count(
        feature_table=regular_features,
        cluster_counts=effective_cluster_range,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )

    mapping_regular = pd.DataFrame(
        {
            "user_id": regular_features.index,
            "cluster_id": labels.astype(int),
            "is_outlier": False,
        }
    )
    mapping_outliers = pd.DataFrame(
        {
            "user_id": feature_table.index[outlier_flags],
            "cluster_id": -1,
            "is_outlier": True,
        }
    )
    mapping_df = pd.concat([mapping_regular, mapping_outliers], ignore_index=True).sort_values(
        ["cluster_id", "user_id"]
    )
    mapping_df.reset_index(drop=True, inplace=True)

    summary_df = summarize_clusters(feature_table, mapping_df)
    mapping_df = mapping_df.merge(summary_df[["cluster_id", "cluster_name"]], on="cluster_id", how="left")
    selection_df["n_outliers_removed"] = int(outlier_flags.sum())
    selection_df["n_regular_users"] = int((~outlier_flags).sum())

    periods = compute_equal_size_periods(test_df.index, n_periods=n_periods)
    protocol = {
        "main_metric": "MAPE_0_100",
        "mape_definition": "Computed on timestamps with strictly positive actual values only.",
        "outlier_handling": {
            "method": "IsolationForest",
            "contamination": outlier_contamination,
            "n_outliers_removed_before_clustering": int(outlier_flags.sum()),
        },
        "clustering_features": _feature_columns(feature_table),
        "fixed_split": {
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "validation_start": str(validation_df.index.min()),
            "validation_end": str(validation_df.index.max()),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
        },
        "equal_size_test_periods": periods_to_records(periods),
    }

    feature_table.to_csv(output_path / "cluster_feature_table.csv")
    selection_df.to_csv(output_path / "cluster_model_selection.csv", index=False)
    mapping_df.to_csv(output_path / "user_cluster_mapping.csv", index=False)
    summary_df.to_csv(output_path / "cluster_summary.csv", index=False)

    with open(output_path / "evaluation_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    return {
        "n_users": int(feature_table.shape[0]),
        "n_outliers": int(outlier_flags.sum()),
        "best_k": int(best_k),
        "output_dir": str(output_path),
        "test_periods": periods_to_records(periods),
    }
