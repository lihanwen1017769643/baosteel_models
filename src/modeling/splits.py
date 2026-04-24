from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def temporal_group_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    group_col: str = "point_name_hint",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    parts = []
    for _, g in df.sort_values([group_col, "监控时间"]).groupby(group_col):
        n = len(g)
        t1 = int(n * train_ratio)
        t2 = int(n * (train_ratio + val_ratio))
        gg = g.copy()
        gg["split"] = "test"
        gg.iloc[:t1, gg.columns.get_loc("split")] = "train"
        gg.iloc[t1:t2, gg.columns.get_loc("split")] = "val"
        parts.append(gg)

    out = pd.concat(parts, ignore_index=True)
    train_df = out[out["split"] == "train"].copy()
    val_df = out[out["split"] == "val"].copy()
    test_df = out[out["split"] == "test"].copy()

    note = {
        "strategy": "按点位内时间顺序切分 train/val/test",
        "reason": "避免随机切分破坏时间依赖，同时覆盖所有点位的时段分布。",
    }
    return train_df, val_df, test_df, note


def leave_one_point_out_split(df: pd.DataFrame, group_col: str = "point_name_hint") -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    point_counts = df[group_col].value_counts()
    holdout_point = point_counts.index[0]
    train_df = df[df[group_col] != holdout_point].copy()
    test_df = df[df[group_col] == holdout_point].copy()
    return train_df, test_df, holdout_point
