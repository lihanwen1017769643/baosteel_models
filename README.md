# 工业时序异常识别原型系统

基于钢铁集团 CEMS 排放监测分钟级报表数据，构建可复现、可扩展的工业时序异常识别 pipeline。

## 项目结构

```
├── configs/
│   └── default.yaml          # 全局配置（数据路径、清洗参数、模型列表等）
├── data/
│   └── 数据报表_20260405/     # 原始 .xls 报表（git ignored）
├── src/
│   ├── config.py              # 配置加载
│   ├── data_ingestion/        # Excel 解析 + 元信息提取
│   ├── data_processing/       # 清洗、标签构造
│   ├── features/              # 时序特征工程
│   ├── modeling/              # 模型训练、评估、切分
│   ├── visualization/         # EDA 图表 + 特征重要性 + 阈值可视化
│   ├── reporting/             # Markdown 报告生成
│   └── utils/                 # IO、日志工具
├── scripts/
│   └── run_pipeline.py        # 一键运行全流程
├── outputs/
│   ├── tables/                # CSV/JSON 统计结果
│   ├── figures/               # 所有图表
│   ├── models/                # 最优模型 + 指标
│   └── cache/parsed/          # Excel 解析缓存
├── docs/
│   └── technical_report.md    # 自动生成的技术报告
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 安装依赖
python3 -m pip install -r requirements.txt

# 运行全流程（解析全部数据）
python3 scripts/run_pipeline.py

# 复用解析缓存（加速重跑）
python3 scripts/run_pipeline.py --resume-parse

# 限制文件数（快速调试）
python3 scripts/run_pipeline.py --max-files 10 --resume-parse
```

## Pipeline 步骤

1. **Excel 解析**: 自动识别多行表头、展平字段名、提取元信息
2. **工艺映射**: 关联排口编号与治理工艺
3. **时间清洗**: 解析时间、排序、去重
4. **标签构造**: 基于设备维护标记构造 anomaly/maintenance 标签
5. **缺失处理**: 分位数裁剪 + 按点位时序 ffill/bfill + 中位数兜底
6. **EDA**: 分布图、缺失率图、标签分布、时序样例、异常窗口、相关性热力图
7. **特征工程**: 时间特征 + lag/diff/rate + rolling(mean/std/min/max/amp) + 超阈值计数
8. **时间切分**: 按点位内时间顺序 70/15/15 切分
9. **模型训练**: LogReg / RandomForest / LightGBM / HistGBM
10. **评估**: Precision/Recall/F1/ROC-AUC/PR-AUC + 阈值优化 + 特征重要性
11. **留一点位泛化**: Leave-one-point-out 验证
12. **报告**: 自动生成 `docs/technical_report.md`

## 输出

- `outputs/tables/`: 数据概览、缺失率、标签分布、特征清单、模型结果、训练日志
- `outputs/figures/`: EDA 图、PR/ROC 曲线、混淆矩阵、特征重要性、阈值权衡
- `outputs/models/`: 最优模型文件 + 指标
- `docs/technical_report.md`: 完整技术报告
