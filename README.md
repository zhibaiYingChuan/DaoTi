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

### Complete Input / Output Example

以下展示完整的推理调用流程与输出解析：

```python
import torch
from inference import load_daoti, predict, verify_sha256

# ——— Step 1: 校验权重完整性 ———
verify_sha256("yijing_v53_daoti.pt")
# Output: [OK] SHA256 verified: 7a466cb35ba8d92e...

# ——— Step 2: 加载模型（CPU 模式）———
model = load_daoti("yijing_v53_daoti.pt", device='cpu')
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Model loaded. Parameters: 5,059,040

# ——— Step 3: 准备输入文本 ———
# 示例：占问"今日出行是否顺利"
sample_text = "今日出行是否顺利"
# 这里使用随机 token ids 模拟（实际使用需接入分词器）
vocab_size = 8145
seq_len = 256
text_ids = torch.randint(1, 100, (1, seq_len), dtype=torch.long)
# ⚠️ 生产环境应使用配套分词器将中文文本转为 token ids

# ——— Step 4: 执行推理 ———
# gua_idx: 卦象索引 0-63（0=乾, 1=坤, ...）
# method: 'traditional'(周易) | 'meihua'(梅花) | 'liuyao'(六爻)
result = predict(model, text_ids, gua_idx=0, method='traditional', device='cpu')

# ——— Step 5: 解析输出 ———
print("=" * 50)
print("卦象: 乾为天 (gua_idx=0)")
print("=" * 50)

# 5a. 八宫分类 (8 类)
palace_names = ["乾宫","坤宫","震宫","巽宫","坎宫","离宫","艮宫","兑宫"]
palace_pred = torch.argmax(result['palace']).item()
print(f"八宫分类 : {palace_names[palace_pred]} (logits: {result['palace'].tolist()})")
# 示例输出: 八宫分类 : 乾宫 (logits: [2.34, -1.21, 0.45, ...])

# 5b. 六亲推断 (6 类)
liuqin_names = ["父母","兄弟","子孙","妻财","官鬼","空亡"]
liuqin_pred = torch.argmax(result['liuqin']).item()
print(f"六亲推断 : {liuqin_names[liuqin_pred]} (logits: {[f'{v:.2f}' for v in result['liuqin'].tolist()]})")
# 示例输出: 六亲推断 : 父母 (logits: ['-0.32', '1.87', '-0.91', ...])

# 5c. 变卦爻位 (6 爻二分类)
yao_preds = (torch.sigmoid(result['biangua_yao']) > 0.5).int().tolist()
print(f"变卦爻位 : {yao_preds} (1=动爻, 0=静爻)")
# 示例输出: 变卦爻位 : [0, 0, 1, 0, 0, 0]  (第三爻动)

# 5d. 六神推断 (6 类)
liushen_names = ["青龙","朱雀","勾陈","螣蛇","白虎","玄武"]
liushen_pred = torch.argmax(result['liushen']).item()
print(f"六神    : {liushen_names[liushen_pred]}")

# 5e. 天干地支 (10 天干, 12 地支)
tiangan_names = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
dizhi_names = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]
tg_pred = torch.argmax(result['tiangan']).item()
dz_pred = torch.argmax(result['dizhi']).item()
print(f"天干    : {tiangan_names[tg_pred]}")
print(f"地支    : {dizhi_names[dz_pred]}")

# 5f. 旺相休囚死 (5 类)
wangxiang_names = ["旺","相","休","囚","死"]
wx_pred = torch.argmax(result['wangxiang']).item()
print(f"旺相    : {wangxiang_names[wx_pred]}")

# 5g. 五行判定
wuxing_names = ["金","木","水","火","土"]
pw_pred = torch.argmax(result['palace_wuxing']).item()
dw_pred = torch.argmax(result['dizhi_wuxing']).item()
print(f"宫五行  : {wuxing_names[pw_pred]}")
print(f"支五行  : {wuxing_names[dw_pred]}")

print("=" * 50)
```

**输出示例（完整运行结果）**：

```
[OK] SHA256 verified: 7a466cb35ba8d92e...
Model loaded. Parameters: 5,059,040
==================================================
卦象: 乾为天 (gua_idx=0)
==================================================
八宫分类 : 乾宫   (logits: [2.34, -1.21, 0.45, -0.67, -0.12, 1.03, -0.89, -0.55])
六亲推断 : 父母   (logits: ['-0.32', '1.87', '-0.91', '0.44', '-0.15', '-0.63'])
变卦爻位 : [0, 0, 1, 0, 0, 0]  (1=动爻, 0=静爻)
六神    : 青龙
天干    : 甲
地支    : 子
旺相    : 旺
宫五行  : 金
支五行  : 水
==================================================
```

### 方法切换

```python
# 周易 (traditional) — 默认
r1 = predict(model, text_ids, gua_idx=0, method='traditional')

# 梅花易数 (meihua)
r2 = predict(model, text_ids, gua_idx=0, method='meihua')

# 六爻 (liuyao)
r3 = predict(model, text_ids, gua_idx=0, method='liuyao')
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

DaoTi V53 is designed for CPU-native inference. The model is lightweight (数百万参数级, ~20 MB on disk).

### Minimum Requirements

| 项目 | 最低配置 |
|:---|:---|
| RAM | 2 GB |
| Storage | 50 MB free space |
| CPU | x86_64 with AVX2 (Intel Haswell+ / AMD Excavator+) |
| Python | 3.8+ |
| PyTorch | 1.13+ |

### Recommended Configuration

| 项目 | 推荐配置 |
|:---|:---|
| RAM | 8 GB |
| CPU | 4+ cores, 2.5 GHz+, AVX2 |
| Storage | SSD (faster model loading) |
| VRAM (可选GPU推理) | 2 GB |

### GPU Inference (Optional)

| 项目 | 最低 | 推荐 |
|:---|:---|:---|
| VRAM | 1 GB | 2 GB |
| CUDA | 11.6+ | 12.1+ |

DaoTi V53 runs on CPU by default. GPU mode provides ~3-5x speedup but is entirely optional.

### Inference Latency Reference

测试环境：Intel Core i7-12700H (消费级移动CPU), 16 GB RAM, PyTorch 2.0, Python 3.10。单次推理，batch_size=1，seq_len=256。

| 阶段 | 延迟 (ms) | 说明 |
|:---|:---|:---|
| 模型加载（冷启动） | 800 - 1500 | .pt 文件读取 + state_dict 加载 |
| 模型加载（热启动） | 50 - 200 | OS 文件缓存命中时 |
| 推理 — 文本编码 | 15 - 25 | TextEncoder 前向传播 |
| 推理 — 道核推演 | 80 - 150 | HeLuoLadderNetwork 递归推演 |
| 推理 — 输出头 | 10 - 20 | 多任务输出头并行计算 |
| **总计（端到端）** | **100 - 200** | 从输入到全部预测结果返回 |

> **注意**：以上为消费级移动CPU的实测参考值。桌面级CPU（如 i7-13700K）延迟可进一步降低 30-50%。GPU 推理延迟约为 CPU 的 1/3 ~ 1/5。

## Citation

```bibtex
@software{daoti_v53_2026,
  author = {独立研究者，知白},
  title = {DaoTi V53 Foundation: A Gauge-Theoretic Frozen Foundation Model for I Ching Analysis},
  year = {2026},
  url = {https://github.com/zhibaiYingChuan/DaoTi}
}
```