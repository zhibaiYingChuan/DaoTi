# DaoTi V53 Foundation Model (道体基座)

> **算力不是门槛。结构性增效，而非规模堆积。**

YiJing V53 Foundation (道体基座) is a neural network model for **I Ching (易经)** divination, text understanding, and multi-domain semantic analysis. It features a **frozen DaoTi + lightweight adapter** paradigm based on the discovery of a **Degenerate Ground State** — a gauge-field-theoretic structure in deep learning. The entire V53 model was trained on a **consumer-grade CPU**, without GPU clusters.

## What's Included

| 文件 | 说明 |
|:---|:---|
| `yijing_v53_daoti.pt` | 模型权重文件 (state_dict，纯数据) |
| `yijing_v53_daoti.pt.sha256` | SHA256 校验文件 |
| `yijing_v53_config.json` | 模型配置参数 |
| `inference.py` | 极简推理脚本 (加载 + 预测) |
| `白皮书_道体基座技术.md` | 技术白皮书 |
| `papers/` | 6 篇研究论文 |

**注意：架构源码、训练代码、训练数据配方不在此仓库中。** 它们作为核心工艺受到保护。详见 [LICENSE](LICENSE)。

## Why DaoTi — Structural Efficiency vs. Scale

| 维度 | 主流大模型 | 道体基座 |
|:---|:---|:---|
| 训练硬件 | 数千GPU集群 | **消费级CPU** |
| 基座参数量 | 数百M ~ 数百B | **数百万级** |
| 适配参数量 | 全量微调 (100%) | **极小比例参数** |
| 多域扩展 | 灾难性遗忘 | **冻结基座，无限扩展** |
| 安全机制 | 外挂护栏 (RLHF) | **架构内置** |

**算力门槛不是自然规律——它是特定技术路线的约束。换一条路线，门槛就不存在了。**

## Quick Start

```python
import torch
from inference import load_daoti, predict, verify_sha256

# 1. Verify integrity
verify_sha256("yijing_v53_daoti.pt")

# 2. Load model
model = load_daoti("yijing_v53_daoti.pt")

# 3. Run prediction
text_ids = torch.randint(1, vocab_size, (1, seq_len))  # your tokenized text
result = predict(model, text_ids, gua_idx=0, method='traditional')
print(result['palace'])   # 八宫分类
print(result['liuqin'])   # 六亲推断
print(result['biangua_yao'])  # 变卦爻位
```

## Key Features

- **CPU-trained**: 数百万参数级模型在消费级CPU上训练完成
- **结构性增效**: 以数量级更低的计算资源达到大参数模型的效果
- **双通道融合**: 文本理解 + 易经符号推理
- **冻结道体**: 核心编码器权重为不变量（退化基态）
- **多任务符号推理**: 易经专业任务平均准确率极高
- **多种占法**: 周易、梅花、六爻
- **卦象检索**: 文本到卦象的原型空间映射

## 保护策略

| 层级 | 内容 | 状态 |
|:---|:---|:---|
| 理论层 | 退化基态理论、规范场论结构、不动点性质 | ✅ 论文+白皮书 已公开 |
| 产品层 | 模型权重 (.pt)、推理接口 | ✅ 本仓库 |
| 工艺层 | 架构源码、训练代码、数据配方 | 🔒 保护中（合作获取） |

**理论确立学术优先权。产品广泛分发建立生态。工艺严格保护形成壁垒。**

## Papers

| # | 论文 | 主题 |
|:---|:---|:---|
| 01 | 退化基态：深度学习中的规范场论结构 | Degenerate Ground State & Gauge Field Structure |
| 02 | 规范场论平行：形式化证明 | Gauge Field Theory Formal Proof |
| 03 | 从规范对称到语义升维 | From Gauge Symmetry to Semantic Dimension Elevation |
| 04 | 从分类到生成：语言涌现动力学 | Language Emergence Dynamics |
| 05 | 从数据炼制到不动点发现 | From Data Refinement to Fixed Point Discovery |
| 06 | 宇宙第一定论 | Cosmic First Principle |
| — | 道体基座技术白皮书 | Technical White Paper |

## 合作获取

如需获取架构源码、训练代码或数据配方进行深度合作，请联系：

- **独立研究者，知白**
- Email: spring60@vip.qq.com
- Website: sfang.cc
- GitHub: https://github.com/zhibaiYingChuan/DaoTi

## License

本仓库中的模型权重、推理脚本和文档依据 **DaoTi Research License v1.0**（中英双语）发布。

架构源码、训练代码等核心工艺不在此仓库中，需另行授权。详见 [LICENSE](LICENSE)。

这不是 MIT、Apache 或 GPL。它是为道体知识产权分层保护而设计的自定义许可证。

## Hardware Requirements

| 需求 | 最低配置 |
|:---|:---|
| RAM | 2 GB |
| VRAM (GPU推理) | 1 GB |
| CPU推理 | 支持 (建议 6+ GB RAM) |

## Citation

```bibtex
@software{daoti_v53_2026,
  author = {独立研究者，知白},
  title = {DaoTi V53 Foundation: A Gauge-Theoretic Frozen Foundation Model for I Ching Analysis},
  year = {2026},
  url = {https://github.com/zhibaiYingChuan/DaoTi}
}
```