# DaoTi V53 Foundation Model (道体基座)

> **Research Preview — Reference Implementation Only**
> 研究预览 — 仅含参考实现
>
> **算力不是门槛。结构性增效，而非规模堆积。**

YiJing V53 Foundation (道体基座) is a neural network model for **I Ching (易经)** divination, text understanding, and multi-domain semantic analysis. It features a **frozen DaoTi + lightweight adapter** paradigm based on the discovery of a **Degenerate Ground State** — a gauge-field-theoretic structure in deep learning. The entire V53 model was trained on a **consumer-grade CPU**, without GPU clusters — a direct challenge to the "no GPU, no foundation model" orthodoxy.

---

## IMPORTANT — Read Before Use | 使用前必读

```
┌─────────────────────────────────────────────────────────────┐
│  THIS REPOSITORY CONTAINS:                                  │
│  ✅ Reference Code (model architecture)                     │
│  ✅ Documentation (whitepaper, 6 papers)                    │
│  ✅ Configuration file                                      │
│  ✅ Architecture verification script                        │
│                                                             │
│  THIS REPOSITORY DOES NOT CONTAIN:                          │
│  ❌ Model Weights (safetensors, .pt, etc.)                  │
│  ❌ SHA256 verification files                               │
│  ❌ Training code                                           │
│                                                             │
│  Model weights require separate written authorization.      │
│  See LICENSE Section 2.2 for access procedures.             │
└─────────────────────────────────────────────────────────────┘
```

See **[LICENSE](LICENSE)** (中英双语) for full terms.

---

## Overview | 概述

| Item | Value |
|:---|:---|
| Architecture | YiJing V53 Foundation (Dual-pathway Bilateral Ladder Network) |
| Parameters | 5,059,040 (5.06M) |
| Public Assets | Reference Code + Documentation |
| Protected Assets | Model Weights (逐案授权) |
| License | DaoTi Research License (see LICENSE) |
| Framework | PyTorch 2.x |

---

## What's Public vs. Protected | 公开与保护

### Layer 1: Theory (Fully Public · 完全公开)

The theoretical foundations are published openly to establish academic priority:
- Degenerate Ground State existence proof
- Gauge field theory structure in deep learning
- Fixed Point discovery and properties
- Multi-domain extension principles

**Assets**: 6 research papers + technical whitepaper

### Layer 2: Architecture (Public with Attribution · 署名公开)

The model architecture design is publicly available as reference code:
- TrigramSpace design (Yin-Yang bifurcator, Wuxing curvature, Bagua sphere)
- HeLuo Ladder Network structure
- Module interfaces and data flow

**Assets**: `yijing_v53_model.py`, `yijing_v53_config.json`

### Layer 3: Weights (Protected · 逐案授权)

Model weights are the core asset. They are NOT publicly distributed.

| Applicant Type | License | Fee |
|:---|:---|:---|
| 中国大陆学术机构 | Academic Use Agreement | Free |
| 中国大陆商业实体 | Commercial License | Negotiated |
| 中国大陆境外实体 | Case-by-case review | Case-by-case |

### Layer 4: Adapters & Ecosystem (Future · 规划中)

- Adapter interface specifications (to be open-sourced)
- Basic demo adapters (to be open-sourced)
- High-value domain adapters (commercial license)

---

## Why DaoTi — Structural Efficiency vs. Scale | 结构性增效 vs 规模堆积

The DaoTi paradigm is not "a smaller LLM." It is a **structurally different approach** that achieves what large models do with orders of magnitude less compute:

| Dimension | Mainstream LLMs | DaoTi V53 |
|:---|:---|:---|
| Training Hardware | Thousands of GPUs | **Consumer CPU** |
| Base Parameters | 100M ~ 100B+ | **~5M** |
| Adapter Training | Full fine-tuning (100%) | **0.67% parameters** |
| Multi-domain Extension | Catastrophic forgetting | **Frozen base, unlimited** |
| Safety Mechanism | External guardrails (RLHF) | **Architecture-native** |
| Training Data | Internet-scale (TB) | **45.9M classical text** |

**Key insight**: The compute barrier is not a law of nature — it is a constraint of a specific technical path. Switch paths, and the barrier disappears.

---

## Architecture | 架构

```
Input Text → TextEncoder (4-head Transformer, 128-dim)
    → TextProj (128→176, FIXED POINT)         ← DAOTI (frozen)
    → HeLuoLadderNetwork (6 layers × 7 steps)  ← DAOTI (frozen)
    → SymbolicInput fusion (gated)             ← DAOTI (frozen)
    → OutputHeadV38 × 3 methods                ← LoRA-adaptable
    → Palace, Tiangan, Dizhi, Liuqin, Liushen, Wangxiang, Biangua_yao
```

**Key Innovation**: The text_proj layer exhibits a **Fixed Point** property — its weights are invariant to perturbations and do not require gradient updates. This enables the *frozen DaoTi + lightweight adapter* paradigm: the core encoder is frozen, and only small LoRA adapters are trained for downstream tasks.

---

## Quick Start | 快速验证

Verify the architecture integrity (no weights required):

```python
python verify_release.py
```

Or manually:

```python
from yijing_v53_model import YiJingV53Foundation, TrigramSpace

model = YiJingV53Foundation(vocab_size=8145)
# Total: ~5,060,000 parameters (random initialization)

import torch
text_ids = torch.randint(1, 8145, (1, 256))
symbol_x = torch.zeros(1, 176)
method_idx = torch.tensor([0], dtype=torch.long)
gua_idx = torch.tensor([0], dtype=torch.long)

with torch.no_grad():
    outputs = model(symbol_x, text_ids, method_idx, gua_idx)
# Architecture verified — random weights only, no real inference
```

**To run actual inference**, obtain authorized weights and use:

```python
from yijing_v53_model import load_v53_model

model = load_v53_model("yijing_v53_foundation.safetensors")
# Requires separately authorized weights. See LICENSE Section 2.2.
```

---

## Key Features

- **CPU-trained foundation**: Entire 5.06M model trained on consumer CPU — no GPU cluster required
- **Structural efficiency**: Achieves what 100M+ parameter models do with orders of magnitude less compute
- **Dual-pathway fusion**: Text understanding + Symbolic I Ching reasoning
- **Frozen DaoTi**: Core encoder weights are invariant (Degenerate Ground State)
- **7-task symbolic reasoning**: 99.96% average accuracy on I Ching professional tasks
- **Multi-method support**: Zhouyi (周易), Meihua (梅花), Liuyao (六爻)
- **Gua retrieval**: Text-to-hexagram mapping via 64-dimensional prototype space
- **TrigramSpace**: Yin-Yang bifurcation, Wuxing curvature, Bagua sphere mapping

---

## Papers | 论文

| # | Paper | Topic |
|:---|:---|:---|
| 01 | [退化基态：深度学习中的规范场论结构](papers/01_退化基态_规范场论结构.md) | Degenerate Ground State & Gauge Field Structure |
| 02 | [规范场论平行：形式化证明](papers/02_规范场论平行_形式化证明.md) | Gauge Field Theory Formal Proof |
| 03 | [从规范对称到语义升维](papers/03_从规范对称到语义升维.md) | From Gauge Symmetry to Semantic Dimension Elevation |
| 04 | [从分类到生成：语言涌现动力学](papers/04_从分类到生成_语言涌现动力学.md) | From Classification to Generation: Language Emergence |
| 05 | [从数据炼制到不动点发现](papers/05_从数据炼制到不动点发现.md) | From Data Refinement to Fixed Point Discovery |
| 06 | [宇宙第一定论](papers/06_宇宙第一定论.md) | Cosmic First Principle |
| — | [道体基座技术白皮书](白皮书_道体基座技术.md) | DaoTi Foundation Technical White Paper |

---

## Getting Model Weights | 获取模型权重

Model weights are protected under the DaoTi Research License and require separate authorization.

### For academic institutions in Chinese mainland (中国大陆学术机构):
Submit an Academic Use Agreement to 独立研究者，知白, including:
- Institution name and accreditation
- Research scope and expected duration
- Responsible PI name and contact

→ Free of charge

### For commercial entities in Chinese mainland (中国大陆商业实体):
Submit a Commercial License Application. Terms negotiated per use case.

### For entities outside Chinese mainland (中国大陆境外实体):
Submit a detailed proposal. Reviewed case-by-case with export control compliance.

---

## License | 许可证

**DaoTi Research License v1.0** (中英双语)

- **Reference Code & Documentation**: Public research license (attribution required, share-alike)
- **Model Weights**: Case-by-case authorization (NOT included in this repository)
- **Export Control**: Subject to PRC export control laws

See **[LICENSE](LICENSE)** for full terms.

This is NOT MIT, Apache, or GPL. It is a custom license designed for the layered protection of the DaoTi intellectual property stack.

---

## Security | 安全

- **Safetensors format**: Trained weights use safetensors (pure data, no code execution risk)
- **SHA256 verification**: Available with authorized weight distribution
- **No pickle loading in public API**: The reference `load_v53_model()` uses safetensors only

---

## File Listing | 文件清单

```
daoti_v53_release/
├── yijing_v53_model.py              # Model architecture (reference implementation)
├── yijing_v53_config.json           # Architecture configuration
├── verify_release.py                # Architecture verification (no weights needed)
├── README.md                        # This file
├── LICENSE                          # DaoTi Research License (中英双语)
├── .gitignore
├── 白皮书_道体基座技术.md             # Technical White Paper (Chinese)
└── papers/                          # Research papers
    ├── 01_退化基态_规范场论结构.md
    ├── 02_规范场论平行_形式化证明.md
    ├── 03_从规范对称到语义升维.md
    ├── 04_从分类到生成_语言涌现动力学.md
    ├── 05_从数据炼制到不动点发现.md
    └── 06_宇宙第一定论.md
```

---

## Hardware Requirements

| Requirement | Minimum |
|:---|:---|
| RAM | 2 GB |
| VRAM (GPU inference) | 1 GB |
| CPU inference | Yes (6+ GB RAM recommended) |

---

## Citation | 引用

```
@software{daoti_v53_2026,
  author = {独立研究者，知白},
  title = {DaoTi V53 Foundation: A Gauge-Theoretic Frozen Foundation Model for I Ching Analysis},
  year = {2026},
  url = {https://github.com/zhibaiYingChuan/DaoTi}
}
```