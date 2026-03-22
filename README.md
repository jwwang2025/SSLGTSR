## SSLGTSR (Reproduction Scaffold)

本仓库提供论文 **“Social Recommendation with Graph Neural Networks Integrating Transformer and Self-Supervised Learning”** 的一个**可运行复现工程骨架**（PyTorch 实现），包含：

- **数据**：用户-物品交互图（UI graph）与用户-用户社交图（UU graph）的加载与稀疏图构建
- **模型**：两路 LightGCN 编码（UI 与 UU）+ **Transformer 融合**（跨视图自注意力）
- **训练**：BPR 主损失 + 图增强 **对比学习自监督损失（InfoNCE）**
- **评测**：HR@K、NDCG@K（Top-K 推荐）
- **可跑**：自带 toy 数据生成器，可端到端 smoke run

> 说明：由于 PDF 文本在当前环境下不可直接抽取检索，本实现对齐了该类论文最常用的“GNN + Transformer + SSL”范式，并把关键组件做成可插拔模块；你拿到论文精确公式后，可以在 `sslgtsr/models/` 下替换对应模块实现以 1:1 对齐。

### 目录结构

```
SSLGTSR/
  configs/
    default.yaml
  scripts/
    make_toy_data.py
  sslgtsr/
    __init__.py
    cli.py
    data/
      __init__.py
      dataset.py
      graph.py
      sampling.py
    models/
      __init__.py
      lightgcn.py
      transformer_fusion.py
      ssl.py
      sslgtsr.py
    training/
      __init__.py
      losses.py
      metrics.py
      trainer.py
    utils/
      __init__.py
      config.py
      logging.py
      seed.py
  train.py
  evaluate.py
  requirements.txt
```

### 安装

建议使用 Python 3.10+（本项目仅依赖 PyTorch / NumPy / SciPy / PyYAML）。

```bash
pip install -r requirements.txt
```

### 1) 准备数据

数据格式采用最常见的 “txt 三列/两列”：

- `data/interactions.txt`: 每行 `user_id item_id`（空格或 tab 分隔），隐式反馈（出现即正样本）
- `data/social.txt`: 每行 `user_id user_id`（无向或有向均可，配置里可指定）

也可以先用 toy 数据跑通：

```bash
python scripts/make_toy_data.py --out_dir data/toy --n_users 200 --n_items 300
```

### 2) 训练

```bash
python train.py --config configs/default.yaml --data_dir data/toy
```

### 3) 评测

```bash
python evaluate.py --config configs/default.yaml --data_dir data/toy --ckpt runs/latest/model.pt
```

### 常见对齐点（你从论文里核对）

- **图编码器**：是否是 LightGCN / GAT / GraphSAGE；如果不是，替换 `sslgtsr/models/lightgcn.py`
- **Transformer**：融合对象是（UI-view, UU-view）还是多层多头跨层 token；替换 `sslgtsr/models/transformer_fusion.py`
- **自监督任务**：本文实现的是“跨视图对比 + 图增强”；若论文是 mask-attribute / neighborhood prediction 等，替换 `sslgtsr/models/ssl.py`
- **训练目标**：BPR vs CE；替换 `sslgtsr/training/losses.py`


