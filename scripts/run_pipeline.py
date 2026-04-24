from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_config
from src.data_ingestion.excel_parser import load_process_mapping, parse_all_excels
from src.data_processing.cleaning import (
    build_label,
    cast_numeric,
    clean_missing_and_outlier,
    infer_numeric_columns,
    to_datetime_and_sort,
)
from src.features.feature_engineering import add_time_features, add_window_features, select_base_numeric_columns
from src.modeling.evaluate import choose_threshold, compute_metrics, plot_confusion, plot_curves
from src.modeling.splits import leave_one_point_out_split, temporal_group_split
from src.modeling.trainers import predict_scores, train_model
from src.reporting.report import build_markdown_report
from src.utils.io_utils import ensure_dir, write_json
from src.utils.logging_utils import setup_logger
from src.visualization.eda import save_basic_eda_tables, save_eda_figures


def main() -> None:
    logger = setup_logger()
    cfg = load_config(ROOT / "configs/default.yaml")

    data_root = ROOT / cfg["data"]["data_root"]
    mapping_file = ROOT / cfg["data"]["mapping_file"]

    out_tables = ensure_dir(ROOT / "outputs/tables")
    out_fig = ensure_dir(ROOT / "outputs/figures")
    ensure_dir(ROOT / "outputs/models")

    logger.info("1) 解析 Excel")
    raw_df = parse_all_excels(
        data_root,
        cfg["data"]["file_glob"],
        max_files=cfg["data"].get("max_files"),
    )
    if raw_df.empty:
        raise RuntimeError("没有解析到可用数据")

    logger.info("2) 关联点位-工艺映射")
    mapping_df = load_process_mapping(mapping_file)
    df = raw_df.merge(mapping_df[["outlet_id", "治理工艺"]], on="outlet_id", how="left")
    df["治理工艺"] = df["治理工艺"].fillna(df.get("process_hint"))

    logger.info("3) 时间与重复清洗")
    df, clean_stats = to_datetime_and_sort(df)

    logger.info("4) 识别数值字段与类型转换")
    numeric_cols = infer_numeric_columns(df, min_non_na_ratio=cfg["cleaning"]["min_non_na_ratio_for_numeric"])
    df = cast_numeric(df, numeric_cols)

    logger.info("5) 标签构造")
    df, label_summary, label_rule = build_label(
        df,
        normal_values=cfg["labeling"]["normal_values"],
        maintenance_values=cfg["labeling"]["maintenance_values"],
    )

    logger.info("6) 缺失与异常值处理")
    ql, qh = cfg["cleaning"]["clip_quantiles"]
    df, missing_stats = clean_missing_and_outlier(df, numeric_cols, ql, qh)

    logger.info("7) EDA 输出")
    save_basic_eda_tables(df, out_tables)
    core_numeric_for_eda = [c for c in numeric_cols if c in df.columns][:12]
    save_eda_figures(df, core_numeric_for_eda, out_fig)
    missing_stats.to_csv(out_tables / "missing_ratio_all_columns.csv", index=False, encoding="utf-8-sig")

    logger.info("8) 特征工程")
    df = add_time_features(df)
    base_numeric_cols = select_base_numeric_columns(df, max_count=cfg["features"]["max_base_numeric_features"])
    fe_res = add_window_features(df, base_numeric_cols, windows=cfg["features"]["windows"])
    feat_df = fe_res.df
    fe_res.feature_catalog.to_csv(out_tables / "feature_engineering_catalog.csv", index=False, encoding="utf-8-sig")

    # 模型字段
    id_cols = {
        "监控时间",
        "label_anomaly",
        "label_is_maintenance",
        "source_file",
        "source_path",
        "split",
        "title_start_time",
        "title_end_time",
        "mapping_outlet_name",
    }
    raw_feature_cols = [
        c
        for c in feat_df.columns
        if c not in id_cols and not c.startswith("title_") and not c.endswith("_date")
    ]
    allowed_cats = {"point_name_hint", "outlet_id", "治理工艺", "source_folder", "process_hint", "生产设施工况"}
    feature_cols = []
    for c in raw_feature_cols:
        if pd.api.types.is_numeric_dtype(feat_df[c]):
            feature_cols.append(c)
        elif c in allowed_cats:
            feature_cols.append(c)

    logger.info("9) 时间切分")
    train_df, val_df, test_df, split_note = temporal_group_split(
        feat_df,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        group_col="point_name_hint",
    )

    max_train_rows = int(cfg["models"].get("max_train_rows", len(train_df)))
    max_val_rows = int(cfg["models"].get("max_val_rows", len(val_df)))
    if len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=cfg["random_state"]).sort_values("监控时间")
    if len(val_df) > max_val_rows:
        val_df = val_df.sample(max_val_rows, random_state=cfg["random_state"]).sort_values("监控时间")

    logger.info("10) 多模型训练与评估")
    experiments = []
    best_art = None
    best_row = None

    for model_name in cfg["models"]["enabled"]:
        art, y_val, val_score, status = train_model(
            model_name,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col="label_anomaly",
            random_state=cfg["random_state"],
        )
        if status != "ok" or art is None:
            experiments.append({"model_name": model_name, "status": status})
            continue

        tsel = choose_threshold(y_val, val_score, min_precision=cfg["models"]["threshold_min_precision"])
        test_score = predict_scores(art, test_df, feature_cols)
        met = compute_metrics(test_df["label_anomaly"].astype(int).to_numpy(), test_score, tsel["threshold"])
        met.update({"model_name": model_name, "status": status})
        experiments.append(met)

        # 保存曲线与混淆矩阵
        plot_curves(test_df["label_anomaly"].astype(int).to_numpy(), test_score, str(out_fig / f"{model_name}"))
        plot_confusion(met, str(out_fig / f"{model_name}_confusion.png"))

        if best_row is None or met["f1"] > best_row["f1"]:
            best_row = met
            best_art = art

    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv(out_tables / "model_results.csv", index=False, encoding="utf-8-sig")

    if best_art is not None:
        joblib.dump(best_art.pipeline, ROOT / "outputs/models/best_model.joblib")
        write_json(best_row, ROOT / "outputs/models/best_model_metrics.json")

    logger.info("11) 留一点位泛化评估")
    if feat_df["point_name_hint"].nunique() >= 2:
        loo_train, loo_test, holdout_point = leave_one_point_out_split(feat_df)
        if len(loo_train) > 0 and len(loo_test) > 0:
            loo_val = loo_test.sample(min(len(loo_test), 5000), random_state=cfg["random_state"])
            loo_art, y_val, val_score, status = train_model(
                "random_forest",
                train_df=loo_train,
                val_df=loo_val,
                feature_cols=feature_cols,
                target_col="label_anomaly",
                random_state=cfg["random_state"],
            )
            if status == "ok" and loo_art is not None:
                hold_score = predict_scores(loo_art, loo_test, feature_cols)
                hold_met = compute_metrics(loo_test["label_anomaly"].astype(int).to_numpy(), hold_score, 0.5)
                hold_met["holdout_point"] = holdout_point
                write_json(hold_met, ROOT / "outputs/tables/leave_one_point_out_metrics.json")
        else:
            write_json({"status": "skipped", "reason": "empty_train_or_test_after_holdout"}, ROOT / "outputs/tables/leave_one_point_out_metrics.json")
    else:
        write_json({"status": "skipped", "reason": "need_at_least_two_points"}, ROOT / "outputs/tables/leave_one_point_out_metrics.json")

    # 数据与字段清单
    fields_df = pd.DataFrame({"field": feat_df.columns, "dtype": [str(feat_df[c].dtype) for c in feat_df.columns]})
    fields_df.to_csv(out_tables / "field_catalog.csv", index=False, encoding="utf-8-sig")

    cleaned_summary = pd.DataFrame(
        {
            "rows": [len(feat_df)],
            "points": [feat_df["point_name_hint"].nunique()],
            "processes": [feat_df["治理工艺"].nunique() if "治理工艺" in feat_df.columns else 0],
            "anomaly_ratio": [feat_df["label_anomaly"].mean()],
        }
    )
    cleaned_summary.to_csv(out_tables / "cleaned_sample_summary.csv", index=False, encoding="utf-8-sig")

    # 报告
    overview = {
        "file_count": int(feat_df["source_file"].nunique()),
        "point_count": int(feat_df["point_name_hint"].nunique()),
        "process_count": int(feat_df["治理工艺"].nunique() if "治理工艺" in feat_df.columns else 0),
        "rows": int(len(feat_df)),
    }

    assumptions = [
        "将状态位字段中 '-' 与空值视为未触发异常；非正常文本统一视为异常。",
        "使用点位内时间切分替代随机切分，以减少时序泄漏。",
        "窗口特征基于分钟粒度顺序生成，跨文件按点位连续拼接。",
        "若XGBoost环境不可用则自动跳过，不影响主流程。",
    ]

    ok_exp_df = exp_df[exp_df["status"] == "ok"].copy() if "status" in exp_df.columns else exp_df
    build_markdown_report(
        report_path=ROOT / "docs/technical_report.md",
        overview=overview,
        clean_stats=clean_stats,
        label_summary=label_summary,
        label_rule=label_rule,
        split_note=split_note,
        experiments=ok_exp_df,
        feature_catalog_path="outputs/tables/feature_engineering_catalog.csv",
        assumptions=assumptions,
    )

    # 训练日志
    run_log = {
        "overview": overview,
        "clean_stats": clean_stats,
        "label_summary": label_summary,
        "split_note": split_note,
        "models": experiments,
    }
    write_json(run_log, ROOT / "outputs/tables/training_log.json")

    logger.info("完成：结果已输出到 outputs/ 和 docs/technical_report.md")


if __name__ == "__main__":
    main()
