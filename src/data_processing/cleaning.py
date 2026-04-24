from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


STATE_COLUMNS_HINT = ["设备维护标记", "维护", "状态"]


def to_datetime_and_sort(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    out["监控时间"] = pd.to_datetime(out.get("监控时间"), errors="coerce")
    invalid_time_rows = int(out["监控时间"].isna().sum())
    out = out.dropna(subset=["监控时间"])

    before_dup = len(out)
    out = out.sort_values(["point_name_hint", "监控时间"]).drop_duplicates(
        subset=["point_name_hint", "监控时间", "source_file"], keep="first"
    )
    dup_removed = before_dup - len(out)

    return out, {"invalid_time_rows": invalid_time_rows, "duplicate_rows_removed": int(dup_removed)}


def infer_numeric_columns(df: pd.DataFrame, min_non_na_ratio: float = 0.05) -> List[str]:
    numeric_cols = []
    for c in df.columns:
        if c in {"监控时间", "source_file", "source_path"}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_non_na_ratio:
            numeric_cols.append(c)
    return numeric_cols


def cast_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_label(df: pd.DataFrame, normal_values: List[str], maintenance_values: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    out = df.copy()
    state_cols = [c for c in out.columns if any(k in c for k in STATE_COLUMNS_HINT)]

    normal_set = {str(x).strip() for x in normal_values}
    maintenance_set = {str(x).strip() for x in maintenance_values}

    def _single_state_abnormal(x: object) -> bool:
        if pd.isna(x):
            return False
        s = str(x).strip()
        if s in normal_set:
            return False
        return True

    if state_cols:
        abnormal = np.zeros(len(out), dtype=bool)
        maintenance = np.zeros(len(out), dtype=bool)
        for c in state_cols:
            vals = out[c]
            abnormal = abnormal | vals.map(_single_state_abnormal).fillna(False).values
            maintenance = maintenance | vals.astype(str).str.strip().isin(maintenance_set).fillna(False).values
        out["label_anomaly"] = abnormal.astype(int)
        out["label_is_maintenance"] = maintenance.astype(int)
    else:
        out["label_anomaly"] = 0
        out["label_is_maintenance"] = 0

    ratio = float(out["label_anomaly"].mean()) if len(out) else 0.0
    summary = {
        "rows": int(len(out)),
        "anomaly_rows": int(out["label_anomaly"].sum()),
        "anomaly_ratio": ratio,
        "has_class_imbalance": bool(ratio < 0.2 or ratio > 0.8),
    }

    rule = (
        "标签构造规则: 优先使用状态位字段(列名包含设备维护标记/维护/状态)。"
        "若任一关键状态字段取值不在正常集合{正常(N), 正常, N, -, 空}中，则label_anomaly=1。"
        "其中维护集合{维护(M), M, 维护}额外记录到label_is_maintenance。"
        "若无状态字段，则默认全部正常(0)。"
    )
    return out, summary, rule


def clean_missing_and_outlier(
    df: pd.DataFrame,
    numeric_cols: List[str],
    q_low: float,
    q_high: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    missing_stats = pd.DataFrame(
        {
            "column": out.columns,
            "missing_ratio": [out[c].isna().mean() for c in out.columns],
            "dtype": [str(out[c].dtype) for c in out.columns],
        }
    ).sort_values("missing_ratio", ascending=False)

    for c in numeric_cols:
        s = out[c]
        if s.notna().sum() < 20:
            continue
        low, high = s.quantile(q_low), s.quantile(q_high)
        out[c] = s.clip(lower=low, upper=high)

    # 分组时间序列插值策略：按点位先ffill/bfill，再用全局中位数兜底
    for c in numeric_cols:
        out[c] = out.groupby("point_name_hint")[c].transform(lambda x: x.ffill().bfill())
        out[c] = out[c].fillna(out[c].median())

    return out, missing_stats
