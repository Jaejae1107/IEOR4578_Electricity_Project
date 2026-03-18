from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .pipeline import _feature_columns


SVG_WIDTH = 960
SVG_HEIGHT = 560


def _cluster_color(cluster_id: int) -> str:
    palette = {
        -1: "#D62728",
        0: "#1F77B4",
        1: "#2CA02C",
        2: "#FF7F0E",
        3: "#9467BD",
        4: "#8C564B",
        5: "#E377C2",
        6: "#7F7F7F",
        7: "#BCBD22",
        8: "#17BECF",
    }
    return palette.get(cluster_id, "#333333")


def _scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def _svg_header(title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="40" y="42" font-size="24" font-family="Arial, sans-serif" fill="#111111">{escape(title)}</text>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _feature_frame(feature_table: pd.DataFrame) -> pd.DataFrame:
    if "user_id" in feature_table.columns:
        feature_frame = feature_table.copy()
    else:
        index_name = feature_table.index.name or "index"
        feature_frame = feature_table.reset_index().rename(columns={index_name: "user_id"})

    if "user_id" not in feature_frame.columns:
        first_col = feature_frame.columns[0]
        feature_frame = feature_frame.rename(columns={first_col: "user_id"})

    return feature_frame


def write_cluster_size_svg(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    width, height = SVG_WIDTH, SVG_HEIGHT
    left, right, top, bottom = 90, 40, 80, 140
    plot_w = width - left - right
    plot_h = height - top - bottom

    bars = summary_df.sort_values("cluster_id").reset_index(drop=True)
    max_users = max(float(bars["n_users"].max()), 1.0)
    bar_slot = plot_w / max(len(bars), 1)
    bar_w = bar_slot * 0.55

    svg = _svg_header("Cluster Size Overview")
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>')

    for tick in range(6):
        value = max_users * tick / 5
        y = _scale(value, 0, max_users, height - bottom, top)
        svg.append(f'<line x1="{left-6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#555" stroke-width="1"/>')
        svg.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#444">{int(round(value))}</text>'
        )

    for idx, row in bars.iterrows():
        cluster_id = int(row["cluster_id"])
        cx = left + (idx + 0.5) * bar_slot
        bar_left = cx - bar_w / 2
        bar_top = _scale(float(row["n_users"]), 0, max_users, height - bottom, top)
        bar_height = (height - bottom) - bar_top
        color = _cluster_color(cluster_id)
        label = f"C{cluster_id}" if cluster_id >= 0 else "Outlier"

        svg.append(
            f'<rect x="{bar_left:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" height="{bar_height:.1f}" '
            f'fill="{color}" opacity="0.9"/>'
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{bar_top-10:.1f}" text-anchor="middle" font-size="13" '
            f'font-family="Arial, sans-serif" fill="#111">{int(row["n_users"])}</text>'
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#222">{escape(label)}</text>'
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{height-bottom+44}" text-anchor="middle" font-size="11" '
            f'font-family="Arial, sans-serif" fill="#666">{escape(str(row["cluster_name"]))}</text>'
        )

    svg.extend(_svg_footer())
    Path(output_path).write_text("\n".join(svg), encoding="utf-8")


def write_cluster_daily_profile_svg(
    feature_table: pd.DataFrame,
    mapping_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    width, height = SVG_WIDTH, SVG_HEIGHT
    left, right, top, bottom = 90, 200, 80, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    merged = mapping_df.merge(_feature_frame(feature_table), on="user_id", how="left")
    hour_cols = [f"hour_profile_{hour:02d}" for hour in range(24)]
    profile_df = merged.groupby(["cluster_id", "cluster_name"], as_index=False)[hour_cols].mean()

    y_min = min(float(profile_df[hour_cols].min().min()), 0.0)
    y_max = max(float(profile_df[hour_cols].max().max()), 1.0)
    y_min = min(0.0, y_min * 0.95)
    y_max = y_max * 1.05

    svg = _svg_header("Average Daily Load Profile by Cluster")
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>')

    for hour in range(24):
        x = _scale(hour, 0, 23, left, width - right)
        if hour < 23:
            svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#EEE" stroke-width="1"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="11" '
            f'font-family="Arial, sans-serif" fill="#444">{hour}</text>'
        )

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = _scale(value, y_min, y_max, height - bottom, top)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#F1F1F1" stroke-width="1"/>')
        svg.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#444">{value:.2f}</text>'
        )

    legend_y = top + 10
    for _, row in profile_df.sort_values("cluster_id").iterrows():
        cluster_id = int(row["cluster_id"])
        color = _cluster_color(cluster_id)
        points = []
        for hour in range(24):
            value = float(row[f"hour_profile_{hour:02d}"])
            x = _scale(hour, 0, 23, left, width - right)
            y = _scale(value, y_min, y_max, height - bottom, top)
            points.append(f"{x:.1f},{y:.1f}")
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}" opacity="0.95"/>'
        )
        svg.append(f'<rect x="{width-right+10}" y="{legend_y-10}" width="16" height="16" fill="{color}" />')
        svg.append(
            f'<text x="{width-right+34}" y="{legend_y+3}" font-size="12" font-family="Arial, sans-serif" '
            f'fill="#222">C{cluster_id if cluster_id >= 0 else "Out"}: {escape(str(row["cluster_name"]))}</text>'
        )
        legend_y += 28

    svg.extend(_svg_footer())
    Path(output_path).write_text("\n".join(svg), encoding="utf-8")


def write_cluster_pca_svg(
    feature_table: pd.DataFrame,
    mapping_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    width, height = SVG_WIDTH, SVG_HEIGHT
    left, right, top, bottom = 90, 90, 80, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    used_columns = _feature_columns(feature_table)
    X = feature_table[used_columns].fillna(0.0).to_numpy()
    X_scaled = RobustScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    scatter_df = mapping_df.merge(
        pd.DataFrame(coords, index=feature_table.index, columns=["pc1", "pc2"]).reset_index().rename(
            columns={"index": "user_id"}
        ),
        on="user_id",
        how="left",
    )

    x_min = float(scatter_df["pc1"].min())
    x_max = float(scatter_df["pc1"].max())
    y_min = float(scatter_df["pc2"].min())
    y_max = float(scatter_df["pc2"].max())

    svg = _svg_header("Cluster Separation (PCA Projection)")
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>')

    for _, row in scatter_df.iterrows():
        x = _scale(float(row["pc1"]), x_min, x_max, left, width - right)
        y = _scale(float(row["pc2"]), y_min, y_max, height - bottom, top)
        color = _cluster_color(int(row["cluster_id"]))
        radius = 6 if bool(row["is_outlier"]) else 4
        opacity = 0.95 if bool(row["is_outlier"]) else 0.65
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}" />'
        )

    centroid_df = scatter_df.groupby(["cluster_id", "cluster_name"], as_index=False)[["pc1", "pc2"]].mean()
    for _, row in centroid_df.iterrows():
        x = _scale(float(row["pc1"]), x_min, x_max, left, width - right)
        y = _scale(float(row["pc2"]), y_min, y_max, height - bottom, top)
        label = f"C{int(row['cluster_id'])}" if int(row["cluster_id"]) >= 0 else "Out"
        svg.append(f'<rect x="{x-18:.1f}" y="{y-12:.1f}" width="36" height="20" rx="4" fill="#FFFFFF" stroke="#333"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{y+3:.1f}" text-anchor="middle" font-size="11" '
            f'font-family="Arial, sans-serif" fill="#111">{escape(label)}</text>'
        )

    var_1 = pca.explained_variance_ratio_[0] * 100.0
    var_2 = pca.explained_variance_ratio_[1] * 100.0
    svg.append(
        f'<text x="{left}" y="{height-20}" font-size="12" font-family="Arial, sans-serif" fill="#444">'
        f'PC1 explains {var_1:.1f}% variance, PC2 explains {var_2:.1f}% variance.</text>'
    )
    svg.extend(_svg_footer())
    Path(output_path).write_text("\n".join(svg), encoding="utf-8")


def write_index_html(summary_df: pd.DataFrame, output_dir: str | Path) -> None:
    summary_rows = []
    for _, row in summary_df.sort_values("cluster_id").iterrows():
        summary_rows.append(
            f"<tr><td>{escape(str(row['cluster_id']))}</td><td>{escape(str(row['cluster_name']))}</td>"
            f"<td>{int(row['n_users'])}</td><td>{float(row['cluster_mean_load']):.2f}</td>"
            f"<td>{escape(str(row['cluster_interpretation']))}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Clustering Visual Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 32px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Clustering Visual Summary</h1>
  <p>This page summarizes the current train-only user clustering results.</p>

  <h2>Cluster Sizes</h2>
  <img src="cluster_size_bar.svg" alt="Cluster size bar chart" />

  <h2>Average Daily Profiles</h2>
  <img src="cluster_daily_profile.svg" alt="Cluster daily profile chart" />

  <h2>PCA Scatter</h2>
  <img src="cluster_pca_scatter.svg" alt="Cluster PCA scatter plot" />

  <h2>Cluster Summary Table</h2>
  <table>
    <thead>
      <tr><th>Cluster</th><th>Name</th><th>Users</th><th>Mean Load</th><th>Interpretation</th></tr>
    </thead>
    <tbody>
      {"".join(summary_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    Path(output_dir, "index.html").write_text(html, encoding="utf-8")


def generate_clustering_visualizations(
    feature_table_path: str | Path,
    mapping_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_table = pd.read_csv(feature_table_path, index_col=0)
    feature_table.index.name = "user_id"
    mapping_df = pd.read_csv(mapping_path)
    summary_df = pd.read_csv(summary_path)

    write_cluster_size_svg(summary_df, output_path / "cluster_size_bar.svg")
    write_cluster_daily_profile_svg(feature_table, mapping_df, output_path / "cluster_daily_profile.svg")
    write_cluster_pca_svg(feature_table, mapping_df, output_path / "cluster_pca_scatter.svg")
    write_index_html(summary_df, output_path)
