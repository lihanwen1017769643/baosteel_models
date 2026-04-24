from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringResult:
    df: pd.DataFrame
    generated_features: List[str]
    dropped_features: List[str]
    feature_catalog: pd.DataFrame
    catalog_detail: List[Dict] = field(default_factory=list)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["监控时间"].dt.hour
    out["dayofweek"] = out["监控时间"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["day_of_month"] = out["监控时间"].dt.day
    return out


def add_window_features(
    df: pd.DataFrame,
    base_numeric_cols: List[str],
    windows: List[int],
    group_col: str = "point_name_hint",
    threshold_count_windows: List[int] | None = None,
) -> FeatureEngineeringResult:
    out = df.copy().sort_values([group_col, "监控时间"]).copy()
    new_cols: Dict[str, pd.Series] = {}
    generated: List[str] = []
    catalog_rows: List[Dict] = []

    g = out.groupby(group_col, group_keys=False)

    for c in base_numeric_cols:
        lag1 = f"{c}_lag1"
        new_cols[lag1] = g[c].shift(1)
        generated.append(lag1)
        catalog_rows.append({"base_feature": c, "derived": lag1, "type": "lag", "window": 1})

        diff1 = f"{c}_diff1"
        new_cols[diff1] = g[c].diff(1)
        generated.append(diff1)
        catalog_rows.append({"base_feature": c, "derived": diff1, "type": "diff", "window": 1})

        rate1 = f"{c}_rate1"
        new_cols[rate1] = new_cols[diff1] / (new_cols[lag1].abs() + 1e-6)
        generated.append(rate1)
        catalog_rows.append({"base_feature": c, "derived": rate1, "type": "change_rate", "window": 1})

        for w in windows:
            roll = g[c].rolling(window=w, min_periods=2)

            mean_col = f"{c}_roll_mean_{w}"
            std_col = f"{c}_roll_std_{w}"
            min_col = f"{c}_roll_min_{w}"
            max_col = f"{c}_roll_max_{w}"
            amp_col = f"{c}_amp_{w}"

            new_cols[mean_col] = roll.mean().reset_index(level=0, drop=True)
            new_cols[std_col] = roll.std().reset_index(level=0, drop=True)
            new_cols[min_col] = roll.min().reset_index(level=0, drop=True)
            new_cols[max_col] = roll.max().reset_index(level=0, drop=True)
            new_cols[amp_col] = new_cols[max_col] - new_cols[min_col]

            for name, ftype in [(mean_col, "rolling_mean"), (std_col, "rolling_std"),
                                (min_col, "rolling_min"), (max_col, "rolling_max"),
                                (amp_col, "amplitude")]:
                generated.append(name)
                catalog_rows.append({"base_feature": c, "derived": name, "type": ftype, "window": w})

    # Threshold-exceeding count: how many times in the window the value exceeded 2-sigma
    if threshold_count_windows:
        for c in base_numeric_cols:
            col_mean = out[c].mean()
            col_std = out[c].std()
            if pd.isna(col_std) or col_std < 1e-9:
                continue
            hi_thresh = col_mean + 2 * col_std
            flag_col = f"_tmp_exceed_{c}"
            out[flag_col] = (out[c] > hi_thresh).astype(float)
            for w in threshold_count_windows:
                cnt_col = f"{c}_exceed_2sigma_cnt_{w}"
                new_cols[cnt_col] = (
                    out.groupby(group_col, group_keys=False)[flag_col]
                    .rolling(window=w, min_periods=1)
                    .sum()
                    .reset_index(level=0, drop=True)
                )
                generated.append(cnt_col)
                catalog_rows.append({"base_feature": c, "derived": cnt_col, "type": "exceed_count_2sigma", "window": w})
            out = out.drop(columns=[flag_col])

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    drop_cols = []
    for c in generated:
        if c in out.columns and out[c].isna().mean() > 0.98:
            drop_cols.append(c)

    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    kept = [c for c in generated if c not in drop_cols]

    catalog = pd.DataFrame(
        {
            "feature": generated,
            "kept": [c in kept for c in generated],
            "drop_reason": ["na_ratio>0.98" if c in drop_cols else "" for c in generated],
        }
    )

    return FeatureEngineeringResult(
        df=out,
        generated_features=kept,
        dropped_features=drop_cols,
        feature_catalog=catalog,
        catalog_detail=catalog_rows,
    )


def select_base_numeric_columns(df: pd.DataFrame, max_count: int = 20) -> List[str]:
    blocked = {
        "label_anomaly",
        "label_is_maintenance",
        "hour",
        "dayofweek",
        "is_weekend",
        "day_of_month",
    }
    candidates = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in blocked and not c.startswith("label_")
    ]

    priority_keywords = ["烟尘", "二氧化硫", "氮氧化物", "废气", "烟气", "氧含量"]
    ordered = sorted(
        candidates,
        key=lambda x: (0 if any(k in x for k in priority_keywords) else 1, x),
    )
    return ordered[:max_count]
