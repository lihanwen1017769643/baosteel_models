from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.io_utils import write_text


def build_markdown_report(
    report_path: Path,
    overview: Dict,
    clean_stats: Dict,
    label_summary: Dict,
    label_rule: str,
    split_note: Dict,
    experiments: pd.DataFrame,
    feature_catalog_path: str,
    assumptions: List[str],
) -> None:
    best_row = experiments.sort_values("f1", ascending=False).iloc[0].to_dict() if len(experiments) else {}

    md = f"""
# 工业时序异常识别技术报告

## 1. 项目背景与目标
本项目基于钢铁集团排放监测分钟级报表数据，构建可复现的工业时序异常识别原型，覆盖解析、清洗、EDA、特征、建模、评估、阈值策略与结果落盘。

## 2. 数据概况
- 文件数: {overview.get('file_count')}
- 点位数: {overview.get('point_count')}
- 工艺数: {overview.get('process_count')}
- 总样本数: {overview.get('rows')}

## 3. 数据清洗与标签构造规则
- 时间字段异常行数: {clean_stats.get('invalid_time_rows')}
- 重复行移除数: {clean_stats.get('duplicate_rows_removed')}
- 异常样本数: {label_summary.get('anomaly_rows')}
- 异常样本占比: {label_summary.get('anomaly_ratio'):.4f}
- 是否类别不平衡: {label_summary.get('has_class_imbalance')}

标签规则说明:
{label_rule}

缺失/异常值处理策略:
- 数值字段统一转数值，无法解析记为缺失
- 对数值字段按分位数裁剪(0.1%~99.9%)处理极端值
- 按点位执行时序 `ffill+bfill`，剩余缺失用列中位数兜底
- 时间字段按点位排序并去重，确保训练数据时间单调

## 4. EDA 发现
详见 `outputs/figures/` 与 `outputs/tables/`：
- 点位/工艺样本量分布
- 字段缺失率
- 标签分布
- 数值字段统计与分布
- 关键因子时序样例与异常窗口样例

## 5. 特征工程设计
- 基础特征: 数值原始变量 + 点位/工艺类别 + 时间特征(`hour/dayofweek/is_weekend`)
- 时序窗口: lag1, diff1, rate1, rolling(mean/std/min/max/amp)
- 窗口长度: 5/10/30/60
- 特征清单: `{feature_catalog_path}`

## 6. 数据集划分策略
- 主策略: {split_note.get('strategy')}
- 选择原因: {split_note.get('reason')}
- 同时补充留一点评估以验证跨点位泛化

## 7. 模型实验设计
- Baseline: Logistic Regression
- 传统模型: Random Forest, HistGradientBoosting, XGBoost(环境可用则启用)
- 指标: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- 阈值: 在验证集上约束最小精度并优化F1

## 8. 结果对比
最佳模型: {best_row.get('model_name', 'N/A')}
- Precision: {best_row.get('precision', float('nan')):.4f}
- Recall: {best_row.get('recall', float('nan')):.4f}
- F1: {best_row.get('f1', float('nan')):.4f}
- ROC-AUC: {best_row.get('roc_auc', float('nan')):.4f}
- PR-AUC: {best_row.get('pr_auc', float('nan')):.4f}
- Threshold: {best_row.get('threshold', float('nan')):.4f}

完整结果见 `outputs/tables/model_results.csv`。

## 9. 阈值与告警策略讨论
本场景关注异常召回，允许一定误报，因此不以 Accuracy 作为主指标。采用PR曲线和阈值扫描后，优先保证最低精度约束，再尽量提升召回与F1，适配保守告警策略。

## 10. 当前局限性
- 历史报表 schema 存在异构，字段语义映射仍可能有残余噪声
- 标签依赖状态位规则，未引入人工复核标签
- 深度时序模型（GRU/LSTM）仅预留扩展接口，当前以传统模型+时序特征为主

## 11. 下一步建议
- 引入按点位自适应阈值与告警抑制逻辑
- 增加跨月份滚动回测与留点位泛化评估
- 补充深度模型实验（GRU/TCN）并与树模型进行稳定性对比
- 与业务侧对齐“非正常状态”字典，细化异常等级

## 附：关键假设
"""
    for i, a in enumerate(assumptions, 1):
        md += f"\n{i}. {a}"

    write_text(md, report_path)
