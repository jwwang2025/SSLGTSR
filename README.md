## SSLGTSR (Social Recommendation with Graph Neural Networks)

本项目实现了论文 **"Social Recommendation with Graph Neural Networks Integrating Transformer and Self-Supervised Learning"** 的完整框架（PyTorch 实现）。

### 核心特性

- **双视角图编码**：用户-物品交互图 (UI Graph) + 用户-用户社交图 (UU Graph)
- **LightGCN 传播**：轻量级图卷积网络，保留多层嵌入用于融合
- **Transformer 融合**：基于注意力采样的跨视图自注意力机制
- **自监督学习**：图增强对比学习 (InfoNCE) + 跨视图对齐
- **拓扑感知位置编码**：最短路径距离、节点度、PageRank

### 目录结构

```
SSLGTSR/
├── configs/                 # 配置文件
│   ├── default.yaml        # 默认配置
│   └── ablation.yaml       # 消融实验配置
├── scripts/                # 数据处理脚本
│   ├── make_toy_data.py   # 生成玩具数据
│   └── prepare_lastfm.py  # 转换 LastFM 数据集
├── sslgtsr/               # 核心代码
│   ├── __init__.py
│   ├── models/            # 模型模块
│   │   ├── sslgtsr.py         # 主模型 (SSLGTSR)
│   │   ├── lightgcn_layer.py  # LightGCN 层
│   │   ├── transformer_layer.py # Transformer 层
│   │   ├── propagation_block.py # 传播块
│   │   ├── fusion_layer.py    # 双视角融合
│   │   ├── attn_sampling.py   # 注意力采样
│   │   ├── topo_pe.py         # 拓扑位置编码
│   │   └── cross_view_ssl.py  # 跨视图 SSL
│   ├── data/              # 数据处理
│   │   ├── dataset.py       # 数据集加载
│   │   ├── graph.py         # 图构建
│   │   └── sampling.py      # BPR 采样
│   ├── training/          # 训练相关
│   │   ├── trainer.py      # 训练器
│   │   ├── losses.py       # 损失函数
│   │   └── metrics.py      # 评估指标
│   └── utils/             # 工具函数
│       ├── config.py      # 配置加载
│       ├── logging.py     # 日志记录
│       └── seed.py        # 随机种子
├── train.py              # 训练入口
├── evaluate.py            # 评估入口
├── search_hyperparams.py # 超参数搜索
└── test_pipeline.py      # 管道测试
```

### 安装

```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. 生成玩具数据

```bash
python scripts/make_toy_data.py --out_dir data/toy --n_users 200 --n_items 300
```

#### 2. 训练模型

```bash
python train.py --config configs/default.yaml --data_dir data/toy
```

#### 3. 评估模型

```bash
python evaluate.py --config configs/default.yaml --data_dir data/toy --ckpt runs/latest/model.pt
```

### 论文公式对照

| 公式 | 描述 | 实现位置 |
|------|------|----------|
| (1)-(3) | 注意力采样 | `attn_sampling.py` |
| (7) | 拓扑位置编码 | `topo_pe.py` |
| (8)-(10) | 多头自注意力 | `transformer_layer.py` |
| (11) | Transformer 层 | `transformer_layer.py` |
| (12)-(14) | LightGCN 传播 | `lightgcn_layer.py` |
| (15) | 双视角融合 | `fusion_layer.py` |
| (16)-(17) | 自监督学习 | `cross_view_ssl.py`, `ssl.py` |

### 配置文件说明

主要超参数（`configs/default.yaml`）：

```yaml
model:
  emb_dim: 64              # 嵌入维度
  n_layers: 4              # 传播层数 K
  transformer:
    n_heads: 2            # 注意力头数
    dropout: 0.1          # Dropout 概率
  attn_sample_size: 15    # 注意力样本数 t
  ssl:
    temperature: 0.2       # InfoNCE 温度参数
    edge_drop_rate: 0.1   # 边 Dropout 率
    feature_drop_rate: 0.1 # 特征 Dropout 率
    weight: 1.0e-5        # SSL 损失权重

train:
  epochs: 50
  batch_size: 1024
  lr: 0.001
  eval_every: 1
  topk: [10, 20]
  early_stopping_patience: 10
```

### 数据格式

- `interactions.txt`: 每行 `user_id item_id`（Tab 分隔）
- `social.txt`: 每行 `user_id user_id`（Tab 分隔）

### 评估指标

- Recall@K
- NDCG@K
