from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
        df["label_anomaly"].value_counts(normalize=True).rename("ratio").reset_index().to_csv(
            out_tables / "label_distribution.csv", index=False, encoding="utf-8-sig"
        )

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        numeric_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_csv(
            out_tables / "numeric_summary.csv", encoding="utf-8-sig"
        )


def save_eda_figures(df: pd.DataFrame, numeric_cols: List[str], out_fig: Path) -> None:
    out_fig.mkdir(parents=True, exist_ok=True)

    # 数据量分布
    fig, ax = plt.subplots(figsize=(10, 5))
    top_points = df["point_name_hint"].value_counts().head(15)
    sns.barplot(x=top_points.values, y=top_points.index, ax=ax)
    ax.set_title("Top Points by Row Count")
    fig.tight_layout()
    fig.savefig(out_fig / "rows_by_point_top15.png", dpi=150)
    plt.close(fig)

    # 缺失率
    miss = df.isna().mean().sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=miss.values, y=miss.index, ax=ax)
    ax.set_title("Top Missing Ratio Columns")
    fig.tight_layout()
    fig.savefig(out_fig / "missing_ratio_top25.png", dpi=150)
    plt.close(fig)

    # 标签分布
    if "label_anomaly" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        dist = df["label_anomaly"].value_counts()
        sns.barplot(x=dist.index.astype(str), y=dist.values, ax=ax)
        ax.set_title("Label Distribution")
        fig.tight_layout()
        fig.savefig(out_fig / "label_distribution.png", dpi=150)
        plt.close(fig)

    # 数值字段分布示例
    for c in numeric_cols[:6]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[c].dropna(), bins=60, kde=False, ax=ax)
        ax.set_title(f"Distribution: {c}")
        fig.tight_layout()
        fig.savefig(out_fig / f"dist_{c}.png", dpi=150)
        plt.close(fig)

    # 时间序列示例
    if numeric_cols:
        sample = df.sort_values("监控时间").groupby("point_name_hint").head(600)
        for c in numeric_cols[:4]:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(data=sample, x="监控时间", y=c, hue="point_name_hint", ax=ax, legend=False)
            ax.set_title(f"Time Series Sample: {c}")
            fig.tight_layout()
            fig.savefig(out_fig / f"ts_sample_{c}.png", dpi=150)
            plt.close(fig)

    # 异常窗口可视化
    if "label_anomaly" in df.columns and numeric_cols:
        anom = df[df["label_anomaly"] == 1].sort_values("监控时间")
        if len(anom):
            pivot_row = anom.iloc[len(anom) // 2]
            point = pivot_row["point_name_hint"]
            t = pivot_row["监控时间"]
            win = df[(df["point_name_hint"] == point) & (df["监控时间"].between(t - pd.Timedelta(minutes=60), t + pd.Timedelta(minutes=60)))]
            fig, ax = plt.subplots(figsize=(12, 4))
            c = numeric_cols[0]
            sns.lineplot(data=win, x="监控时间", y=c, ax=ax)
            ax.axvline(t, color="red", linestyle="--")
            ax.set_title(f"Anomaly Window ({point}) - {c}")
            fig.tight_layout()
            fig.savefig(out_fig / "anomaly_window_example.png", dpi=150)
            plt.close(fig)
