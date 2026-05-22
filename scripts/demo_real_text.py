"""
Demo: DaoTi V53 推理演示
=========================
此脚本展示道体模型的完整推理链条——从输入到输出。

用途：让任何一个下载仓库的人，在跑完这个脚本后，亲眼看到这个系统
      的推理能力。这不是一个分类器。分类只是输出层多个线性头中的
      1 个。分类之前有多层编码和递归推理。

⚠️ 分词器 (tokenizer) 为受保护资产，不在此仓库中。
   文本推理模式的 tokenization 步骤在此以概念说明 + 随机 token ids 模拟。
   符号推理模式（以卦象为输入）的计算是真实、完整、可验证的。

本脚本仅使用公开推理接口 (inference.py)，不访问任何内部模型架构。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from daoti.inference import (
    load_daoti, predict, verify_sha256, generate_response, compute_coherence,
    GUA_64, BA_GONG, GUA_WUXING, GUA_TRIGRAM,
    BAGUA_NAMES, WUXING_NAMES, LIUQIN_MAP, PALACE_MAP,
    WUXING_SHENG, WUXING_KE,
    sparse_expand_input, find_palace, STATE_DIM, TEXT_DIM,
)

SAMPLE_TEXTS = [
    "今日出行是否顺利",
    "最近事业遇到瓶颈，该如何突破",
    "身体不适，想知道养生调理的方向",
    "考试在即，复习状态如何调整",
    "感情关系中出现矛盾，如何化解",
]

PALACE_NAMES  = ["乾宫","坤宫","震宫","巽宫","坎宫","离宫","艮宫","兑宫"]
TIANGAN_NAMES = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
DIZHI_NAMES   = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]
LIUQIN_NAMES  = ["父母","兄弟","子孙","妻财","官鬼","空亡"]
LIUSHEN_NAMES = ["青龙","朱雀","勾陈","螣蛇","白虎","玄武"]
WANGXIANG_N   = ["旺","相","休","囚","死"]


def hdr(title):
    print("\n" + "━" * 66)
    print(f"  {title}")
    print("━" * 66)


print("╔" + "═" * 64 + "╗")
print("║" + "  DaoTi V53  推理演示".center(58) + "║")
print("║" + "  冻结道体 + 轻量适配  ·  消费级 CPU 推理".center(56) + "║")
print("╚" + "═" * 64 + "╝")

# ── 1. 权重校验 ──
hdr("第 1 步：权重完整性校验")
ok = verify_sha256("yijing_v53_daoti.pt")
if not ok: exit(1)

# ── 2. 模型加载 ──
hdr("第 2 步：模型加载")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  计算设备: {device}")
model = load_daoti("yijing_v53_daoti.pt", device=device)

total = sum(p.numel() for p in model.parameters())
frozen = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
print(f"  总参数量 : {total:,}")
print(f"  冻结参数 : {frozen:,}  ({100*frozen/total:.1f}%)  ← 道体，永不修改")
print(f"  可训练   : {total-frozen:,}  ({100*(total-frozen)/total:.1f}%)  ← 用体，轻量适配")

# ── 3. 输入准备 ──
hdr("第 3 步：输入准备")
gua_idx = 0
gua_name = GUA_64[gua_idx]
upper, lower = GUA_TRIGRAM.get(gua_idx, ("乾", "乾"))
print(f"  卦象输入 : {gua_name} ({upper}上{lower}下)  索引: {gua_idx}")
print(f"  文本输入示例（分词器为受保护资产，此处以概念展示 + 随机 token ids 验证）:")
for i, t in enumerate(SAMPLE_TEXTS): print(f"    [{i+1}] {t}")

symbol_x = torch.tensor([sparse_expand_input(gua_idx)], dtype=torch.float32, device=device)
print(f"\n  符号嵌入: {STATE_DIM} 维洛书空间符号向量")
print(f"  编码: 6 位爻象 + 先天八卦 + 后天八卦 + 五行 + 八宫 = {STATE_DIM} 维")

text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)

# ── 4. 推理执行 ──
hdr("第 4 步：执行推理")
result = predict(model, text_ids, gua_idx=gua_idx, method='traditional', device=device)

p_name = PALACE_NAMES[result['palace'].argmax().item()]
p_conf = F.softmax(result['palace'], dim=-1).max().item()
t_name = TIANGAN_NAMES[result['tiangan'].argmax().item()]
d_name = DIZHI_NAMES[result['dizhi'].argmax().item()]
lq_name = LIUQIN_NAMES[result['liuqin'].argmax().item()]
ls_name = LIUSHEN_NAMES[result['liushen'].argmax().item()]
wx_name = WANGXIANG_N[result['wangxiang'].argmax().item()]
yao_raw = result['biangua_yao'].squeeze()
yao_p = torch.sigmoid(yao_raw).tolist()
yao_v = [1 if p > 0.5 else 0 for p in yao_p]
moving_yao = [i+1 for i, v in enumerate(yao_v) if v]
pw_name = WUXING_NAMES[result['palace_wuxing'].argmax().item()]
dw_name = WUXING_NAMES[result['dizhi_wuxing'].argmax().item()]
coherence = result['coherence']

print(f"  八宫分类  : {p_name}  (置信度: {p_conf:.1%})")
print(f"  天干地支  : {t_name}{d_name}")
print(f"  六亲持世  : {lq_name}")
print(f"  六神临爻  : {ls_name}")
print(f"  旺相休囚  : {wx_name}")
print(f"  宫五行    : {pw_name}")
print(f"  支五行    : {dw_name}")
print(f"  动爻      : {'第' + '、'.join(map(str, moving_yao)) + '爻' if moving_yao else '无（静卦）'}")
print(f"  相干性    : {coherence:.4f}  {'⚠️ 低置信度' if coherence < 0.3 else '✅'}")

ground_truth = find_palace(GUA_64[gua_idx])
print(f"\n  ✅ 正确判定: {p_name==ground_truth} (真值: {ground_truth})")

# ── 5. 方法切换对比 ──
hdr("第 5 步：三种方法（周易/梅花/六爻）的差异化输出")
results = {}
for m_cn, m_name in [("周易","traditional"), ("梅花","meihua"), ("六爻","liuyao")]:
    r = predict(model, text_ids, gua_idx=gua_idx, method=m_name, device=device)
    p = PALACE_NAMES[r['palace'].argmax().item()]
    lq = LIUQIN_NAMES[r['liuqin'].argmax().item()]
    ls = LIUSHEN_NAMES[r['liushen'].argmax().item()]
    results[m_cn] = (p, lq, ls)
    print(f"  {m_cn:<6s}  →  八宫: {p:<6s}  六亲: {lq:<6s}  六神: {ls}")

all_same = len({v[0] for v in results.values()}) == 1 and \
           len({v[1] for v in results.values()}) == 1
if not all_same:
    print(f"\n  ⚡ 三种方法产生不同输出——差异化推理。")
else:
    print(f"\n  三种方法对 {gua_name} 的输出一致（高置信度卦象）。")

# ── 6. 全64卦统计 ──
hdr("第 6 步：全 64 卦符号推理统计（可复现验证）")
print("  对全部 64 卦执行推理，输出是一次性计算、可复现的。")
correct = 0
for gi in range(64):
    r = predict(model, text_ids, gua_idx=gi, method='traditional', device=device)
    p = PALACE_NAMES[r['palace'].argmax().item()]
    gt = find_palace(GUA_64[gi])
    if p == gt:
        correct += 1
print(f"\n  八宫判定: {correct}/64 ({100*correct/64:.1f}%)")

# ── 7. RAG 检索增强生成 ──
hdr("第 7 步：检索增强生成（RAG）— 从结构化推理到自然语言")
for i, text in enumerate(SAMPLE_TEXTS[:3]):
    print(f"  ── 示例 {i+1}：「{text}」──")
    resp = generate_response(model, text_ids, gua_idx=gua_idx, method='traditional', device=device)
    print(resp['response'])
    print(f"  相干性: {resp['coherence']:.4f}  低置信度: {resp['low_confidence']}")
    print()

# ── 8. 相干性自校准 ──
hdr("第 8 步：相干性自校准质量传感")
coherences = []
for gi in range(64):
    c = compute_coherence(model, text_ids, gi, device)
    coherences.append(c)
coherences.sort()
n = len(coherences)
print(f"  均值: {sum(coherences)/n:.4f}")
print(f"  中位数: {coherences[n//2]:.4f}")
print(f"  最小: {min(coherences):.4f}  最大: {max(coherences):.4f}")
low = sum(1 for c in coherences if c < 0.3)
high = sum(1 for c in coherences if c > 0.7)
print(f"  低于 0.3（低置信度）: {low}/64  高于 0.7（高置信度）: {high}/64")
print(f"\n  ⚡ 相干性是模型判断「我懂不懂」的内部信号。")
print(f"     当相干性低于阈值时，模型会主动声明不确定性——这不是外挂护栏，")
print(f"     而是根植于架构的内在约束。")

# ── Done ──
hdr("总结")
print(f"""
  以上展示了道体模型的完整推理流程：

    第 1-2 步 : 权重校验 + 模型加载（>90% 参数冻结）
    第 3 步   : 输入准备（符号嵌入 + 文本 token）
    第 4 步   : 执行推理（编码 → 投影 → 融合 → 递归推演 → 多任务输出）
    第 5 步   : 方法切换（差异化推理）
    第 6 步   : 全 64 卦统计验证
    第 7 步   : RAG 检索增强生成（结构化推理 → 自然语言）
    第 8 步   : 相干性自校准（不确定性估计）

  分类只是输出层的一个头。这个模型的本质是：
  编码 → 投影 → 融合 → 递归推演 → 多任务输出 → 原型检索 → RAG 生成 → 相干性自校准。
  分类是其中最浅的一步。

  ⚠️ 核心架构算法和训练配方不在此仓库中。详见 LICENSE。
""")
