# Industrial Time-Series Anomaly Prototype

## Run

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_pipeline.py
```

### Common Run Modes

```bash
# Use parser cache to resume quickly
python3 scripts/run_pipeline.py --resume-parse

# Parse all source files (full dataset)
python3 scripts/run_pipeline.py --full-data --resume-parse

# Override max files / parse progress interval
python3 scripts/run_pipeline.py --max-files 40 --batch-size 10
```

## Outputs

- `outputs/tables/`: 清洗统计、字段清单、特征清单、模型评估结果、训练日志
- `outputs/figures/`: EDA 图、PR/ROC、混淆矩阵等
- `outputs/models/`: 最优模型与指标
- `docs/technical_report.md`: 实验技术报告
