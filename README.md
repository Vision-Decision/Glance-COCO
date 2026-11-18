# Glance-Data

这是 Jev 风格 grounding 任务的**数据仓库**，只管数据。任务形式是：一张图配多条文字需求，每条需求的答案是一个框、多个框，或者“图里没有”。

- **诊断**：找出现有数据（RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO）和 benchmark 的问题；
- **增强**：生成新数据。目标框、数量、某类是否存在都由 COCO 标注精确决定，DeepSeek 看图只负责写句子；
- **验证**：用规则过滤，再做分层人工抽查，检查标签是否正确；
- **产出**：增强后的训练数据，以及一份留出图片上的诊断集，供模型仓库使用。

模型、训练和评测都不在这个仓库里。

## Pipeline

```
图片 + 已有标注 ──► ① inventory（规则）──► ② generate（DeepSeek 看图，每图 1 次）──► ③ assemble（规则）──► 增强数据
   COCO 物体框          合并标注，挑出物体编号        短/中/长、口语、中文、               过滤 + 按标注造题
   已有人工描述                                     细节不对的负例、多目标、按用途        （类别名、全部、数量、位置、相近类别）
```

产出的题型：
- **正例**：短 / 中 / 长描述、口语、中文、只说类别名、全部实例、多个目标、按用途描述。
- **负例**：颜色等细节不对、动作不对、数量不对、位置不对、相近类别。

平均每张图约 95 题，DeepSeek 费用约 0.003 元 / 图。

## 快速开始

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env && set -a && source .env && set +a   # 填好数据路径和 key 文件路径
make test
make analyze                       # 诊断现有数据
make pipeline SPLIT=train          # 增强全部训练图
make diag-set                      # 300 张留出图片的诊断集
make audit SPLIT=train             # 分层抽样出图，人工判断后用 audit score 汇总
```

数据格式、设计约束、验证流程和服务器使用约定见 [AGENTS.md](AGENTS.md)，结果见 [docs/RESULTS.md](docs/RESULTS.md)。

## 结果摘要

- **现有数据的缺口**：每个目标都有同类干扰物，所以从没出现过“只说类别名”“全部”这类说法；负例大多是图里根本没有的类别；6.9 万条多目标描述没有使用；只有英文。
- **标签正确率**：按题型分层人工抽查 112 条，101 条正确（约 90%）。
