from __future__ import annotations

import argparse
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
from src.visualization.eda import (
    plot_feature_importance,
    plot_threshold_tradeoff,
    save_basic_eda_tables,
    save_eda_figures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industrial time-series anomaly pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--max-files", type=int, default=None, help="Override max parsed source files")
    parser.add_argument("--full-data", action="store_true", help="Parse all files under data root")
    parser.add_argument("--batch-size", type=int, default=None, help="Parser progress batch size")
    parser.add_argument("--resume-parse", action="store_true", help="Reuse parser cache when available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    cfg = load_config(ROOT / args.config)

    data_root = ROOT / cfg["data"]["data_root"]
    mapping_file = ROOT / cfg["data"]["mapping_file"]

    out_tables = ensure_dir(ROOT / "outputs/tables")
    out_fig = ensure_dir(ROOT / "outputs/figures")
    ensure_dir(ROOT / "outputs/models")

    # ── 1. Parse Excel ──────────────────────────────────────────────────
    logger.info("1) 解析 Excel 报表")
    max_files = cfg["data"].get("max_files")
    if args.max_files is not None:
        max_files = args.max_files
    if args.full_data:
        max_files = None

    batch_size = int(args.batch_size or cfg["data"].get("parse_batch_size", 20))

    raw_df = parse_all_excels(
        data_root,
        cfg["data"]["file_glob"],
        max_files=max_files,
        batch_size=batch_size,
        cache_dir=ROOT / cfg["data"].get("parse_cache_dir", "outputs/cache/parsed"),
        resume_parse=args.resume_parse,
    )
    if raw_df.empty:
        raise RuntimeError("没有解析到可用数据")
    logger.info(f"   → 解析完成: {raw_df['source_file'].nunique()} 个文件, {len(raw_df):,} 行")

    # ── 2. Merge mapping ────────────────────────────────────────────────
    logger.info("2) 关联点位-工艺映射")
    mapping_df = load_process_mapping(mapping_file)
    df = raw_df.merge(mapping_df[["outlet_id", "治理工艺"]], on="outlet_id", how="left")
    df["治理工艺"] = df["治理工艺"].fillna(df.get("process_hint"))

    # ── 3. Time cleaning ────────────────────────────────────────────────
    logger.info("3) 时间解析、排序与去重")
    df, clean_stats = to_datetime_and_sort(df)
    logger.info(f"   → 无效时间行: {clean_stats['invalid_time_rows']}, 重复移除: {clean_stats['duplicate_rows_removed']}")

    # ── 4. Numeric inference ────────────────────────────────────────────
    logger.info("4) 识别数值字段与类型转换")
    numeric_cols = infer_numeric_columns(df, min_non_na_ratio=cfg["cleaning"]["min_non_na_ratio_for_numeric"])
    df = cast_numeric(df, numeric_cols)
    logger.info(f"   → 识别到 {len(numeric_cols)} 个数值字段")

    # ── 5. Label construction ───────────────────────────────────────────
    logger.info("5) 标签构造")
    df, label_summary, label_rule = build_label(
        df,
        normal_values=cfg["labeling"]["normal_values"],
        maintenance_values=cfg["labeling"]["maintenance_values"],
    )
    logger.info(
        f"   → 异常: {label_summary['anomaly_rows']:,} / {label_summary['rows']:,} "
        f"({label_summary['anomaly_ratio']:.4f}), 类别不平衡: {label_summary['has_class_imbalance']}"
    )

    # ── 6. Missing & outlier cleaning ───────────────────────────────────
    logger.info("6) 缺失值与异常值处理")
    ql, qh = cfg["cleaning"]["clip_quantiles"]
    df, missing_stats = clean_missing_and_outlier(df, numeric_cols, ql, qh)
    missing_stats.to_csv(out_tables / "missing_ratio_all_columns.csv", index=False, encoding="utf-8-sig")

    # ── 7. EDA ──────────────────────────────────────────────────────────
    logger.info("7) EDA 输出")
    save_basic_eda_tables(df, out_tables)
    core_numeric_for_eda = [c for c in numeric_cols if c in df.columns][:12]
    save_eda_figures(df, core_numeric_for_eda, out_fig)
    logger.info(f"   → 图表已输出至 {out_fig}")

    # ── 8. Feature engineering ──────────────────────────────────────────
    logger.info("8) 特征工程")
    df = add_time_features(df)
    base_numeric_cols = select_base_numeric_columns(df, max_count=cfg["features"]["max_base_numeric_features"])
    logger.info(f"   → 基础数值特征: {len(base_numeric_cols)} 个")

    threshold_count_windows = cfg["features"].get("threshold_count_windows")
    fe_res = add_window_features(
        df, base_numeric_cols,
        windows=cfg["features"]["windows"],
        threshold_count_windows=threshold_count_windows,
    )
    feat_df = fe_res.df
    fe_res.feature_catalog.to_csv(out_tables / "feature_engineering_catalog.csv", index=False, encoding="utf-8-sig")
    logger.info(f"   → 生成 {len(fe_res.generated_features)} 个特征, 丢弃 {len(fe_res.dropped_features)} 个")

    if fe_res.catalog_detail:
        pd.DataFrame(fe_res.catalog_detail).to_csv(
            out_tables / "feature_engineering_detail.csv", index=False, encoding="utf-8-sig"
        )

    # ── 9. Select model feature columns ─────────────────────────────────
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
    logger.info(f"   → 最终模型特征: {len(feature_cols)} 个")

    # ── 10. Temporal split ──────────────────────────────────────────────
    logger.info("9) 时间切分")
    train_df, val_df, test_df, split_note = temporal_group_split(
        feat_df,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        group_col="point_name_hint",
    )

    train_val_test_sizes = {
        "训练集 (train)": len(train_df),
        "验证集 (val)": len(val_df),
        "测试集 (test)": len(test_df),
    }
    logger.info(f"   → train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    max_train_rows = int(cfg["models"].get("max_train_rows", len(train_df)))
    max_val_rows = int(cfg["models"].get("max_val_rows", len(val_df)))
    if len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=cfg["random_state"]).sort_values("监控时间")
        logger.info(f"   → 训练集采样至 {max_train_rows:,}")
    if len(val_df) > max_val_rows:
        val_df = val_df.sample(max_val_rows, random_state=cfg["random_state"]).sort_values("监控时间")
        logger.info(f"   → 验证集采样至 {max_val_rows:,}")

    # ── 11. Multi-model training & evaluation ───────────────────────────
    logger.info("10) 多模型训练与评估")
    experiments = []
    best_art = None
    best_row = None

    for model_name in cfg["models"]["enabled"]:
        logger.info(f"   → 训练 {model_name} ...")
        art, y_val, val_score, status = train_model(
            model_name,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col="label_anomaly",
            random_state=cfg["random_state"],
        )
        if status != "ok" or art is None:
            logger.warning(f"   → {model_name}: {status}")
            experiments.append({"model_name": model_name, "status": status})
            continue

        tsel = choose_threshold(y_val, val_score, min_precision=cfg["models"]["threshold_min_precision"])
        test_score = predict_scores(art, test_df, feature_cols)
        y_test = test_df["label_anomaly"].astype(int).to_numpy()
        met = compute_metrics(y_test, test_score, tsel["threshold"])
        met.update({"model_name": model_name, "status": status})
        experiments.append(met)

        logger.info(
            f"   → {model_name}: F1={met['f1']:.4f}, Prec={met['precision']:.4f}, "
            f"Rec={met['recall']:.4f}, ROC-AUC={met['roc_auc']:.4f}, PR-AUC={met['pr_auc']:.4f}"
        )

        plot_curves(y_test, test_score, str(out_fig / f"{model_name}"))
        plot_confusion(met, str(out_fig / f"{model_name}_confusion.png"))
        plot_threshold_tradeoff(y_test, test_score, out_fig, model_name)

        if art.feature_importances:
            plot_feature_importance(art.feature_importances, out_fig, model_name)
            imp_df = pd.DataFrame(
                sorted(art.feature_importances.items(), key=lambda x: -abs(x[1])),
                columns=["feature", "importance"],
            )
            imp_df.to_csv(out_tables / f"feature_importance_{model_name}.csv", index=False, encoding="utf-8-sig")

        if best_row is None or met["f1"] > best_row["f1"]:
            best_row = met
            best_art = art

    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv(out_tables / "model_results.csv", index=False, encoding="utf-8-sig")

    if best_art is not None:
        joblib.dump(best_art.pipeline, ROOT / "outputs/models/best_model.joblib")
        write_json(best_row, ROOT / "outputs/models/best_model_metrics.json")
        logger.info(f"   → 最优模型: {best_row['model_name']} (F1={best_row['f1']:.4f})")

    # ── 12. Leave-one-point-out generalization ──────────────────────────
    logger.info("11) 留一点位泛化评估")
    loo_metrics = None
    if feat_df["point_name_hint"].nunique() >= 2:
        loo_train, loo_test, holdout_point = leave_one_point_out_split(feat_df)
        if len(loo_train) > 0 and len(loo_test) > 0:
            loo_val = loo_test.sample(min(len(loo_test), 5000), random_state=cfg["random_state"])
            loo_art, y_v, v_score, status = train_model(
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
                loo_metrics = hold_met
                write_json(hold_met, ROOT / "outputs/tables/leave_one_point_out_metrics.json")
                logger.info(f"   → LOO holdout={holdout_point}, F1={hold_met['f1']:.4f}")
            else:
                write_json({"status": "skipped", "reason": status}, ROOT / "outputs/tables/leave_one_point_out_metrics.json")
        else:
            write_json({"status": "skipped", "reason": "empty_train_or_test_after_holdout"}, ROOT / "outputs/tables/leave_one_point_out_metrics.json")
    else:
        write_json({"status": "skipped", "reason": "need_at_least_two_points"}, ROOT / "outputs/tables/leave_one_point_out_metrics.json")

    # ── 13. Field & summary catalogs ────────────────────────────────────
    logger.info("12) 输出字段清单与数据摘要")
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

    # ── 14. Technical report ────────────────────────────────────────────
    logger.info("13) 生成技术报告")
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
        "若 LightGBM/XGBoost 环境不可用则自动跳过，不影响主流程。",
        "不同点位的缺失因子（如转炉无 SO2 列）以中位数填补，模型需学习点位差异。",
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
        missing_stats=missing_stats,
        fe_catalog_detail=fe_res.catalog_detail,
        loo_metrics=loo_metrics,
        point_list=sorted(feat_df["point_name_hint"].unique().tolist()),
        process_list=sorted(feat_df["治理工艺"].dropna().unique().tolist()) if "治理工艺" in feat_df.columns else None,
        train_val_test_sizes=train_val_test_sizes,
    )

    # ── 15. Training log ────────────────────────────────────────────────
    run_log = {
        "run_args": {
            "config": args.config,
            "max_files": max_files,
            "full_data": args.full_data,
            "batch_size": batch_size,
            "resume_parse": args.resume_parse,
        },
        "overview": overview,
        "clean_stats": clean_stats,
        "label_summary": label_summary,
        "split_note": split_note,
        "train_val_test_sizes": {k: int(v) for k, v in train_val_test_sizes.items()},
        "models": experiments,
    }
    write_json(run_log, ROOT / "outputs/tables/training_log.json")

    logger.info("=" * 60)
    logger.info("完成：结果已输出到 outputs/ 和 docs/technical_report.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
