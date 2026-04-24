from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid")


def save_basic_eda_tables(df: pd.DataFrame, out_tables: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)

    summary = {
        "file_count": df["source_file"].nunique(),
        "point_count": df["point_name_hint"].nunique(),
        "process_count": df["治理工艺"].nunique() if "治理工艺" in df.columns else 0,
        "rows": len(df),
    }
    pd.DataFrame([summary]).to_csv(out_tables / "dataset_overview.csv", index=False, encoding="utf-8-sig")

    df.groupby("point_name_hint").size().rename("rows").reset_index().to_csv(
        out_tables / "rows_by_point.csv", index=False, encoding="utf-8-sig"
    )

    if "治理工艺" in df.columns:
        df.groupby("治理工艺").size().rename("rows").reset_index().to_csv(
            out_tables / "rows_by_process.csv", index=False, encoding="utf-8-sig"
        )

    if "label_anomaly" in df.columns:
        label_dist = df["label_anomaly"].value_counts(normalize=True).rename("ratio").reset_index()
        label_dist.to_csv(out_tables / "label_distribution.csv", index=False, encoding="utf-8-sig")

        label_by_point = df.groupby("point_name_hint")["label_anomaly"].agg(["sum", "count", "mean"])
        label_by_point.columns = ["anomaly_count", "total", "anomaly_ratio"]
        label_by_point.to_csv(out_tables / "label_by_point.csv", encoding="utf-8-sig")

        if "治理工艺" in df.columns:
            label_by_process = df.groupby("治理工艺")["label_anomaly"].agg(["sum", "count", "mean"])
            label_by_process.columns = ["anomaly_count", "total", "anomaly_ratio"]
            label_by_process.to_csv(out_tables / "label_by_process.csv", encoding="utf-8-sig")

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        numeric_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_csv(
            out_tables / "numeric_summary.csv", encoding="utf-8-sig"
        )


def save_eda_figures(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    out_fig.mkdir(parents=True, exist_ok=True)

    _plot_rows_by_point(df, out_fig)
    _plot_rows_by_process(df, out_fig)
    _plot_missing_ratio(df, out_fig)
    _plot_label_distribution(df, out_fig)
    _plot_label_by_point(df, out_fig)
    _plot_numeric_distributions(df, numeric_cols, out_fig)
    _plot_boxplot_by_point(df, numeric_cols, out_fig)
    _plot_boxplot_by_process(df, numeric_cols, out_fig)
    _plot_correlation_heatmap(df, numeric_cols, out_fig)
    _plot_time_series_samples(df, numeric_cols, out_fig)
    _plot_anomaly_windows(df, numeric_cols, out_fig)


def _plot_rows_by_point(df: pd.DataFrame, out_fig: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(5, df["point_name_hint"].nunique() * 0.5)))
    top_points = df["point_name_hint"].value_counts().head(20)
    sns.barplot(x=top_points.values, y=top_points.index, ax=ax, orient="h")
    ax.set_title("Rows by Point")
    ax.set_xlabel("Row Count")
    fig.tight_layout()
    fig.savefig(out_fig / "rows_by_point_top15.png", dpi=150)
    plt.close(fig)


def _plot_rows_by_process(df: pd.DataFrame, out_fig: Path) -> None:
    if "治理工艺" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, df["治理工艺"].nunique() * 0.6)))
    proc_counts = df["治理工艺"].value_counts()
    sns.barplot(x=proc_counts.values, y=proc_counts.index, ax=ax, orient="h")
    ax.set_title("Rows by Process Type")
    ax.set_xlabel("Row Count")
    fig.tight_layout()
    fig.savefig(out_fig / "rows_by_process.png", dpi=150)
    plt.close(fig)


def _plot_missing_ratio(df: pd.DataFrame, out_fig: Path) -> None:
    miss = df.isna().mean().sort_values(ascending=False).head(30)
    miss = miss[miss > 0]
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(miss) * 0.35)))
    sns.barplot(x=miss.values, y=miss.index, ax=ax, orient="h")
    ax.set_title("Top Missing Ratio Columns")
    ax.set_xlabel("Missing Ratio")
    fig.tight_layout()
    fig.savefig(out_fig / "missing_ratio_top25.png", dpi=150)
    plt.close(fig)


def _plot_label_distribution(df: pd.DataFrame, out_fig: Path) -> None:
    if "label_anomaly" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    dist = df["label_anomaly"].value_counts()
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(dist.index.astype(str), dist.values, color=colors[:len(dist)])
    for bar, v in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:,}", ha="center", va="bottom")
    ax.set_title("Label Distribution (0=Normal, 1=Anomaly)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_fig / "label_distribution.png", dpi=150)
    plt.close(fig)


def _plot_label_by_point(df: pd.DataFrame, out_fig: Path) -> None:
    if "label_anomaly" not in df.columns:
        return
    agg = df.groupby("point_name_hint")["label_anomaly"].mean().sort_values(ascending=False)
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.5)))
    sns.barplot(x=agg.values, y=agg.index, ax=ax, orient="h")
    ax.set_title("Anomaly Ratio by Point")
    ax.set_xlabel("Anomaly Ratio")
    fig.tight_layout()
    fig.savefig(out_fig / "anomaly_ratio_by_point.png", dpi=150)
    plt.close(fig)


def _plot_numeric_distributions(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    cols = [c for c in numeric_cols if c in df.columns][:8]
    if not cols:
        return
    n = len(cols)
    ncols_grid = min(4, n)
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(5 * ncols_grid, 4 * nrows_grid))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, c in enumerate(cols):
        sns.histplot(df[c].dropna(), bins=60, kde=True, ax=axes[i])
        axes[i].set_title(f"{c}")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Feature Distributions", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_fig / "numeric_distributions_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot_by_point(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    cols = [c for c in numeric_cols if c in df.columns][:6]
    if not cols:
        return
    for c in cols:
        fig, ax = plt.subplots(figsize=(12, max(4, df["point_name_hint"].nunique() * 0.5)))
        data = df[["point_name_hint", c]].dropna()
        if data.empty:
            plt.close(fig)
            continue
        sns.boxplot(data=data, x=c, y="point_name_hint", ax=ax, orient="h")
        ax.set_title(f"{c} Distribution by Point")
        fig.tight_layout()
        fig.savefig(out_fig / f"boxplot_by_point_{c}.png", dpi=150)
        plt.close(fig)


def _plot_boxplot_by_process(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    if "治理工艺" not in df.columns:
        return
    cols = [c for c in numeric_cols if c in df.columns][:6]
    if not cols:
        return
    for c in cols:
        fig, ax = plt.subplots(figsize=(12, max(4, df["治理工艺"].nunique() * 0.6)))
        data = df[["治理工艺", c]].dropna()
        if data.empty:
            plt.close(fig)
            continue
        sns.boxplot(data=data, x=c, y="治理工艺", ax=ax, orient="h")
        ax.set_title(f"{c} Distribution by Process")
        fig.tight_layout()
        fig.savefig(out_fig / f"boxplot_by_process_{c}.png", dpi=150)
        plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    cols = [c for c in numeric_cols if c in df.columns][:15]
    if len(cols) < 2:
        return
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), max(7, len(cols) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_fig / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_time_series_samples(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    if not numeric_cols:
        return
    points = df["point_name_hint"].unique()
    sample_points = points[:min(4, len(points))]
    cols = [c for c in numeric_cols if c in df.columns][:4]

    for c in cols:
        fig, axes = plt.subplots(len(sample_points), 1, figsize=(14, 3.5 * len(sample_points)), sharex=False)
        if len(sample_points) == 1:
            axes = [axes]
        for ax, pt in zip(axes, sample_points):
            pt_data = df[df["point_name_hint"] == pt].sort_values("监控时间").head(1000)
            ax.plot(pt_data["监控时间"], pt_data[c], linewidth=0.6, alpha=0.8)
            if "label_anomaly" in pt_data.columns:
                anom = pt_data[pt_data["label_anomaly"] == 1]
                if len(anom) > 0:
                    ax.scatter(anom["监控时间"], anom[c], color="red", s=8, zorder=5, label="Anomaly")
            ax.set_title(f"{c} - {pt}", fontsize=10)
            ax.legend(fontsize=8) if len(pt_data[pt_data.get("label_anomaly", pd.Series()) == 1]) > 0 else None
        fig.suptitle(f"Time Series: {c}", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(out_fig / f"ts_sample_{c}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_anomaly_windows(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    if "label_anomaly" not in df.columns or not numeric_cols:
        return
    anom = df[df["label_anomaly"] == 1].sort_values("监控时间")
    if anom.empty:
        return

    cols = [c for c in numeric_cols if c in df.columns][:3]
    points_with_anom = anom["point_name_hint"].unique()[:3]

    for pidx, point in enumerate(points_with_anom):
        pt_anom = anom[anom["point_name_hint"] == point]
        if pt_anom.empty:
            continue
        pivot_row = pt_anom.iloc[len(pt_anom) // 2]
        t = pivot_row["监控时间"]
        win = df[
            (df["point_name_hint"] == point)
            & (df["监控时间"].between(t - pd.Timedelta(minutes=120), t + pd.Timedelta(minutes=120)))
        ].sort_values("监控时间")

        if win.empty:
            continue

        fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3.5 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, c in zip(axes, cols):
            ax.plot(win["监控时间"], win[c], linewidth=0.8)
            anom_win = win[win["label_anomaly"] == 1]
            if len(anom_win) > 0:
                ax.scatter(anom_win["监控时间"], anom_win[c], color="red", s=12, zorder=5)
            ax.axvline(t, color="red", linestyle="--", alpha=0.5, label="Pivot anomaly")
            ax.set_ylabel(c, fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(f"Anomaly Window: {point}", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(out_fig / f"anomaly_window_{pidx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_feature_importance(importances: dict, out_fig: Path, model_name: str, top_n: int = 25) -> None:
    if not importances:
        return
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, vals = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.35)))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance: {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_fig / f"feature_importance_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_threshold_tradeoff(y_true: np.ndarray, y_score: np.ndarray, out_fig: Path, model_name: str) -> None:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    thresholds_ext = np.append(thresholds, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds_ext, precision, label="Precision", linewidth=1.5)
    ax.plot(thresholds_ext, recall, label="Recall", linewidth=1.5)
    ax.plot(thresholds_ext, f1, label="F1", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 vs Threshold: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig / f"threshold_tradeoff_{model_name}.png", dpi=150)
    plt.close(fig)
