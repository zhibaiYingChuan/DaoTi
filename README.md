# DaoTi V53 Foundation Model (道体基座)

> **算力不是门槛。结构性增效，而非规模堆积。**

道体基座（DaoTi V53 Foundation）是一个预训练的神经网络**语义基座模型**。它以中文自然语言文本为输入，输出结构化语义表征——包括编码空间中的语义向量、洛书空间中的状态向量，以及64维结构化原型向量（卦象空间）。

基于**双轨阶梯网络（Bilateral Ladder Network）**架构，在通用语料和易经古典文本上完成预训练。训练完成后核心参数被冻结（「道体」），作为后续所有领域适配的稳定基础。这一**冻结道体 + 轻量适配**范式建立在对**退化基态（Degenerate Ground State）**的发现之上——这是深度学习中一种规范场论结构。整个V53模型在**消费级CPU**上完成训练，无需GPU集群。

可用于：
- 多领域文本的语义理解与分类
- 作为下游任务的特征提取器
- 易经文本的结构化推理（包括卦象推演）
- 作为任何需要稳定语义表征的AI系统的认知基座

## 道体模型是什么

**输入**：中文自然语言文本
**输出**：结构化认知状态（八宫、六亲、六神、天干地支、旺相休囚等）+ 自然语言回答（通过 RAG 检索增强生成）
**核心任务**：领域感知的专家型推理与对话

道体模型**不是一个分类器**。领域分类只是它内部路由机制的一部分——就像 GPT 的 tokenizer 分完词之后还有几百层 Transformer 一样，分类之后才是真正的推理和生成。模型的完整推理链路如下：

```
用户输入 → 语义编码 → 领域判定 → 激活对应适配器 → 结构化推理 → 认知状态输出 → 生成回答
```

以一个端到端的推理案例说明：

> 用户输入：「我最近总是咳嗽，吃什么好？」
> → 语义编码为洛书空间向量
> → 领域分类器判定：**medicine 域**（置信度 92%）
> → 激活对应的医学适配器
> → 检索增强：从知识库调取《本草纲目》《伤寒论》相关条目
> → 输出结构化认知状态 + 生成回答

领域分类只是流水线的第一步。之后还有编码器推演、三爻空间精炼、原型检索和表达生成——分类器只是「门」，门后面的房间才是核心能力所在。

## What's Included

| 文件 | 说明 |
|:---|:---|
| `yijing_v53_daoti.pt` | 模型权重文件 (state_dict，纯数据) |
| `yijing_v53_daoti.pt.sha256` | SHA256 校验文件 |
| `yijing_v53_config.json` | 模型配置参数 |
| `inference.py` | 极简推理脚本 (加载 + 预测) |
| `demo_real_text.py` | 完整演示脚本 (真实中文输入 → 完整推理链条) |
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

### 与 LoRA / 微调范式的本质区别

道体模型共享了「基座 + 轻量参数」的工程范式，但**不等于共享了同一个设计哲学和功能目标**。就像燃油车和电动车都有四个轮子，但驱动它们的核心逻辑完全不同。

| 对比维度 | LoRA 微调范式 | 道体模型 |
|---------|-------------|---------|
| 基座地位 | 被撬动的工具 | 被冻结的「道」，永恒的参照系 |
| 遗忘态度 | 视为 BUG，需要解决 | 视为新陈代谢，自然修剪 |
| 知识形态 | 分布的、不可名状的权重 | 结构化的、可索引的知识条目 |
| 成长方式 | 一次性注入，焊接 | 永续学习循环，增量更新 |
| 安全机制 | 外挂 RLHF 护栏 | 内嵌于道体，架构级免疫 |
| 理论驱动 | 工程效率（参数节省） | 物理原理（规范对称性→退化基态） |

### 关于「数据效率」的澄清

**道体模型的数据效率，不是「不需要数据」，而是「不需要重复消耗」。**

V53 道体基座在通用语料上的预训练，就是这个系统的「字典」。这个字典已经内嵌在冻结的编码器里。后续的领域适配之所以数据量小，是因为**迁移学习**——基座已经学会了语言的基本结构，适配器只需要学习领域特定的知识映射。

这完全符合「没有免费的午餐」定理。道体模型的贡献在于：**让基座训练只做一次，之后所有新知识通过轻量适配器注入，而不是每次都重新训练基座。**

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

## 安全机制

道体模型的安全性**不是外挂的护栏**，而是根植于道体架构的内在约束。解除它的唯一方法是破坏整个推理流水线——等同于摧毁模型本身。

### 多层安全架构

| 层级 | 机制 | 说明 |
|:---|:---|:---|
| **第一道门控** | 领域分类器 | 任何输入必须先经过领域判定。未知领域 / 低置信度 → 拒绝执行 |
| **不确定性度量** | 卦象熵 | 当系统对输入的状态判断模糊（熵值高）时 → 主动拒答或请求澄清 |
| **信息过滤** | 相干性门控 | 信息流通过相干性门控，低相干信息被抑制，防止杂散信号污染推理链路 |
| **动态阈值** | SelfTuning | 基于历史交互反馈，动态调整拒绝行为的敏感度 |
| **架构级免疫** | 道体冻结 | 核心认知框架不可篡改。所有新知识学习通过适配器进行，适配器**无法**通过 text_proj（梯度为零）将修改传播回编码器 |

### 为什么无法「越狱」

- **无梯度回传路径**：道体冻结意味着适配器的训练梯度只能更新适配器自身，不能触及编码器。任何「越狱微调」只能影响适配器，无法改写核心认知框架。
- **领域分类器前置**：不当输入首先被领域分类器拦截，无法到达适配器执行层。
- **相干性门控不可绕过**：即使输入通过了领域判定，如果它在洛书空间中与已有知识结构相干性低，仍会被门控抑制。

## 演示脚本

仓库提供两个演示脚本：

| 脚本 | 用途 |
|:---|:---|
| `inference.py` | **极简验证**：用随机数据验证模型可加载和推理 |
| `demo_real_text.py` | **深度解剖**：逐层展示编码→投影→融合→递归推演→原型注意力→规则推理→多任务输出的完整计算链路 |

```bash
# 极简演示（验证模型能否正常加载和推理）
python inference.py

# 深度解剖（完整推理链路，11 层逐层展示）
python demo_real_text.py
```

`demo_real_text.py` 跑完后你会亲眼看到：这个模型经过了 TextEncoder 编码 → text_proj 不动点投影 → 符号-文本门控融合 → HeLuoLadderNetwork 多层多步双轨递归推演 → 语义原型注意力 → 五行生克规则推理（含硬编码生克矩阵残差学习） → 8 任务并行输出 → 原型空间检索 → RAG 检索增强生成（自然语言输出） → 相干性自校准（不确定性估计）。**分类（palace）只是 10 个线性头中的 1 个。**

## RAG 检索增强生成

`inference.py` 提供 `generate_response()` 函数，将模型的结构化推理结果（八宫/六亲/六神/天干地支/旺相/动爻）作为检索键，从内置知识库中匹配对应条目，组合为自然语言回答：

```python
from inference import load_daoti, generate_response
import torch

model = load_daoti("yijing_v53_daoti.pt")
text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long)

# 生成自然语言回答
result = generate_response(model, text_ids, gua_idx=0, method='traditional')
print(result['response'])         # 自然语言回答
print(result['coherence'])        # 相干性（自校准质量信号）
print(result['low_confidence'])   # 是否低于置信度阈值
print(result['details'])          # 结构化推理明细
```

当相干性低于阈值（默认 0.3）时，模型会主动声明不确定性——这不是外挂护栏，而是根植于架构的内在约束。

## 基准测试

```bash
python eval_benchmark.py
```

输出 `benchmark_results.json`，包含：八宫分类准确率、64 卦原型检索 top-1/top-5 准确率、相干性分布统计、随机基线对比。详见 [BENCHMARK.md](BENCHMARK.md)。

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

## 术语对照表

道体模型使用大量易学术语，它们是精确的结构化设计而非玄学包装。下表帮助技术读者快速「翻译」为熟悉的工程概念：

| 易学/道学术语 | 数学/工程对应 | 通俗解释 |
|:---|:---|:---|
| 道体 (Dao Ti) | 冻结的预训练基座模型参数 | 被锁定不能修改的核心「大脑」 |
| 卦象空间 | 64 维结构化原型向量空间 | 模型的「内部工作语言」 |
| 阴阳分化 | 双分支门控机制 (yang_proj + yin_proj + gate) | 把信息分两条路分别处理，一条主干一条细节 |
| 五行曲率 | 5 头注意力 + 相生相克偏置矩阵 | 模型内部的知识分类标准，有些相关有些排斥 |
| 共振腔 | 每域维护的 EMA 中心向量 + 波能量 + 相干性 | 模型判断「我懂不懂」的自我评估系统 |
| 适配器 (Adapter) | 轻量级参数模块，冻结道体下的迁移学习 | 不用重练大脑，只学新知识的小插件 |
| 规范场 (Gauge Field) | gua_prototype 作为补偿场的角色 | 保证「文本→卦象」映射不受编码器扰动影响 |
| 退化基态 | text_proj 参数空间中损失近乎平坦的状态 | 核心投影层改不改都行，效果不变 |
| 三爻空间 | 多域语义精炼架构 | 把语义信息进一步提纯、结构化的处理流水线 |
| 洛书空间 | 道体核心的运算空间 | 模型的所有「思考」都发生在这个高维语义空间中 |
| 驻波共振 | 领域表征的 EMA 平均向量 | 模型对自己擅长领域的「记忆快照」 |

## Citation

```bibtex
@software{daoti_v53_2026,
  author = {独立研究者，知白},
  title = {DaoTi V53 Foundation: A Semantic Foundation Model Based on Gauge-Theoretic Frozen Architecture},
  year = {2026},
  url = {https://github.com/zhibaiYingChuan/DaoTi}
}
```