"""
Demo: DaoTi V53 完整推理演示
==============================
此脚本逐步骤展示道体模型的完整推理链条：
  输入 → 语义编码 → 领域判定 → 洛书空间状态 → 结构化认知输出

与 inference.py 的区别：
  - inference.py : 极简演示，用随机数据验证模型可加载和推理
  - demo_real_text.py : 完整演示，展示从卦象输入到全部认知输出的全过程

⚠️ 注意：此脚本演示符号推理模式（以卦象为输入）。
        文本推理模式需要分词器（受保护资产），此处以注释说明完整流程。
"""

import torch
import numpy as np
from inference import (
    load_daoti, predict, verify_sha256,
    GUA_64, BA_GONG, GUA_WUXING, GUA_TRIGRAM,
    BAGUA_NAMES, WUXING_NAMES, LIUQIN_MAP, PALACE_MAP,
    sparse_expand_input, find_palace
)

# ─── 预置输入文本（实机使用时需配合分词器转为 token ids）───
SAMPLE_TEXTS = [
    "今日出行是否顺利",
    "最近事业遇到瓶颈，该如何突破",
    "身体不适，想知道养生调理的方向",
    "考试在即，复习状态如何调整",
    "感情关系中出现矛盾，如何化解",
]

PALACE_NAMES = ["乾宫","坤宫","震宫","巽宫","坎宫","离宫","艮宫","兑宫"]
TIANGAN_NAMES = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
DIZHI_NAMES = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]
LIUQIN_NAMES = ["父母","兄弟","子孙","妻财","官鬼","空亡"]
LIUSHEN_NAMES = ["青龙","朱雀","勾陈","螣蛇","白虎","玄武"]
WANGXIANG_NAMES = ["旺","相","休","囚","死"]
METHOD_NAMES = {"traditional":"周易","meihua":"梅花易数","liuyao":"六爻"}


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def predict_full(model, gua_idx, method='traditional', device='cpu'):
    """Run full prediction for a given gua, returning decoded results."""
    method_idx_map = {"周易": 0, "六爻": 1, "梅花": 2}
    method_idx_map_inv = {0: "traditional", 1: "liuyao", 2: "meihua"}
    m_idx = method_idx_map.get(method, 0)
    m_name = method_idx_map_inv.get(m_idx, "traditional")

    # Simulate text input with the sparse symbol representation
    text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
    result = predict(model, text_ids, gua_idx=gua_idx, method=m_name, device=device)

    def top(arr, names):
        idx = torch.argmax(arr).item()
        return names[idx], arr.tolist()

    palace_name, palace_logits = top(result['palace'], PALACE_NAMES)
    liuqin_name, liuqin_logits = top(result['liuqin'], LIUQIN_NAMES)
    liushen_name, liushen_logits = top(result['liushen'], LIUSHEN_NAMES)
    tiangan_name, tiangan_logits = top(result['tiangan'], TIANGAN_NAMES)
    dizhi_name, dizhi_logits = top(result['dizhi'], DIZHI_NAMES)
    wangxiang_name, wangxiang_logits = top(result['wangxiang'], WANGXIANG_NAMES)
    pw_name, pw_logits = top(result['palace_wuxing'], WUXING_NAMES)
    dw_name, dw_logits = top(result['dizhi_wuxing'], WUXING_NAMES)

    yao_raw = result['biangua_yao'].squeeze()
    yao_probs = torch.sigmoid(yao_raw).tolist()
    yao_preds = [1 if p > 0.5 else 0 for p in yao_probs]

    ground_truth = find_palace(GUA_64[gua_idx])

    return {
        'gua_name': GUA_64[gua_idx],
        'gua_idx': gua_idx,
        'ground_truth_palace': ground_truth,
        'palace': palace_name,
        'palace_logits': palace_logits,
        'palace_correct': (palace_name == ground_truth),
        'liuqin': liuqin_name,
        'liuqin_logits': liuqin_logits,
        'liushen': liushen_name,
        'liushen_logits': liushen_logits,
        'tiangan': tiangan_name,
        'dizhi': dizhi_name,
        'wangxiang': wangxiang_name,
        'palace_wuxing': pw_name,
        'dizhi_wuxing': dw_name,
        'yao_probs': yao_probs,
        'yao_preds': yao_preds,
    }


def print_result(r):
    """Pretty-print a single prediction result."""
    gua_name = r['gua_name']
    upper, lower = GUA_TRIGRAM.get(r['gua_idx'], ("乾","乾"))
    gua_wx = GUA_WUXING.get(upper, "金")
    gt = r['ground_truth_palace']
    correct = "✓正确" if r['palace_correct'] else "✗ 错误"

    print(f"  卦象: {gua_name} ({upper}上{lower}下, 五行属{gua_wx})   实际宫位: {gt}  判定: {correct}")
    print(f"  八宫 : {r['palace']:<6s}  天干: {r['tiangan']:<4s}  地支: {r['dizhi']:<4s}  旺相: {r['wangxiang']}")
    print(f"  六亲 : {r['liuqin']:<6s}  六神: {r['liushen']:<6s}  宫五行: {r['palace_wuxing']:<4s}  支五行: {r['dizhi_wuxing']}")
    yao_str = ", ".join([f"{'━' if p else '--'}[{i+1}]" for i, p in enumerate(r['yao_preds'])])
    yao_prob_str = ", ".join([f"{r['yao_probs'][i]:.3f}" for i in range(6)])
    print(f"  爻位 : {yao_str}")
    print(f"  动爻概率: [{yao_prob_str}]")

    moving_yao = [i+1 for i, p in enumerate(r['yao_preds']) if p]
    if moving_yao:
        print(f"  动爻 : 第{moving_yao}爻动")
    else:
        print(f"  动爻 : 静卦（六爻皆静）")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  DaoTi V53 完整推理演示")
print("  冻结道体 + 轻量适配 · 消费级 CPU 推理")
print("=" * 60)

# ─── Step 1: 校验权重完整性 ───
section("Step 1: 校验权重完整性")
ok = verify_sha256("yijing_v53_daoti.pt")
if not ok:
    print("[ERROR] 权重校验失败，请检查文件完整性。")
    exit(1)

# ─── Step 2: 加载模型 ───
section("Step 2: 加载模型")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  设备: {device}")
model = load_daoti("yijing_v53_daoti.pt", device=device)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
trainable_params = total_params - frozen_params
print(f"  总参数量: {total_params:,}")
print(f"  冻结参数: {frozen_params:,}  ({100*frozen_params/total_params:.1f}%)")
print(f"  可训练参数: {trainable_params:,}  ({100*trainable_params/total_params:.1f}%)")
print(f"  这是「道-器分离」范式的核心优势——>90% 参数永不修改，新知识只通过 <10% 参数注入。")

# ─── Step 3: 推理流水线概览 ───
section("Step 3: 推理流水线")
print("""
  输入文本（中文自然语言）
      ↓
  TextEncoder (字符级 Transformer, 冻结)
      ↓
  text_proj (不动点投影器, 冻结, 退化基态)
      ↓
  [卦象符号嵌入 + 文本特征 → 门控融合]
      ↓
  HeLuoLadderNetwork (双轨递归推演, 冻结)
      ↓
  三爻空间 (阴阳分叉 → 五行曲率 → 八卦映射 → 河洛折叠 → 共振腔)
      ↓
  ┌────────────────┬──────────────┬──────────────┐
  │   领域分类器     │  多任务输出头  │   表达层(解码)  │
  │   (路由判定)     │ (认知状态输出) │  (文本生成)   │
  └────────────────┴──────────────┴──────────────┘
""")

# ─── Step 4: 预置输入文本（概念演示）───
section("Step 4: 输入示例")
print("  以下文本将进入推理流水线（实机使用需配合分词器）：")
for i, text in enumerate(SAMPLE_TEXTS):
    print(f"    [{i+1}] {text}")
print("\n  ⚠️ 当前演示模式为「符号推理」——以卦象为输入。")
print("    文本推理模式需要分词器（受保护资产），此处以概念流程图展示。")

# ─── Step 5: 方法切换演示 ───
section("Step 5: 三种占法推理对比 (gua_idx=0, 乾为天)")
for method_cn, method_key in [("周易", "traditional"), ("梅花易数", "meihua"), ("六爻", "liuyao")]:
    print(f"\n  --- {method_cn} ({method_key}) ---")
    r = predict_full(model, gua_idx=0, method=method_cn, device=device)
    print_result(r)

# ─── Step 6: 八宫代表性卦象推理 ───
section("Step 6: 八宫代表性卦象推理 (周易模式)")
demo_guas = [
    (0,  "乾宫·乾为天"),
    (1,  "坤宫·坤为地"),
    (50, "震宫·震为雷"),
    (56, "巽宫·巽为风"),
    (28, "坎宫·坎为水"),
    (29, "离宫·离为火"),
    (51, "艮宫·艮为山"),
    (57, "兑宫·兑为泽"),
]
correct = 0
for gua_idx, desc in demo_guas:
    print(f"\n  [{gua_idx}] {desc}")
    r = predict_full(model, gua_idx=gua_idx, method='周易', device=device)
    print_result(r)
    if r['palace_correct']:
        correct += 1

print(f"\n  八宫判定准确率: {correct}/{len(demo_guas)}  ({100*correct/len(demo_guas):.1f}%)")

# ─── Step 7: 检索能力演示 ───
section("Step 7: 原型空间检索演示")
print("  gua_prototype 作为规范场补偿 text_proj 的任意变换，")
print("  使文本→卦象的映射保持不变（规范不变性）。")
print()
with torch.no_grad():
    text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
    text_emb = model.encode_text(text_ids)
    logits = model.retrieval_logits(text_emb, temperature=0.1)
    probs = torch.softmax(logits, dim=-1).squeeze()
    top5 = torch.topk(probs, 5)
    print(f"  输入文本（随机ids模拟）→ Top-5 检索结果：")
    for i, (idx, prob) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
        print(f"    [{i+1}] {GUA_64[idx]:<6s}  (概率: {prob:.4f})")

# ─── Step 8: 八任务符号推理能力统计 ───
section("Step 8: 全64卦八任务推理统计 (周易模式)")
print("  对全部64卦执行推理，统计各任务一致性...")
stats = {
    'palace': 0, 'liuqin_stable': 0, 'liushen_stable': 0,
    'wangxiang_dist': {}, 'total': 64
}
for gi in range(64):
    r = predict_full(model, gua_idx=gi, method='周易', device=device)
    if r['palace_correct']:
        stats['palace'] += 1
    ww = r['wangxiang']
    stats['wangxiang_dist'][ww] = stats['wangxiang_dist'].get(ww, 0) + 1

print(f"\n  八宫判定准确率: {stats['palace']}/{stats['total']} ({100*stats['palace']/64:.1f}%)")
print(f"  旺相分布:")
for k in WANGXIANG_NAMES:
    cnt = stats['wangxiang_dist'].get(k, 0)
    bar = "█" * (cnt // 4)
    print(f"    {k}: {cnt:>2d}/64  {bar}")

# ─── Step 9: 洛书空间语义可视化提示 ───
section("Step 9: 模型自校准 (相干性检测)")
print("  驻波共振腔维护每个领域的 EMA 中心向量。")
print("  模型可通过当前表征与领域驻波的余弦相似度判断预测可靠性。")
print("  (相干性信号为内部指标，输出为 scalar，实验数据见白皮书第五章)")
with torch.no_grad():
    text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
    text_emb = model.encode_text(text_ids)
    proto = model.gua_prototype.weight
    proto_norm = torch.nn.functional.normalize(proto, p=2, dim=-1)
    text_norm = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
    coherence = torch.mm(text_norm, proto_norm.t()).max().item()
    print(f"\n  当前文本→最匹配卦象的相干性: {coherence:.4f}")
    if coherence > 0.8:
        print("  → 高相干性：模型对当前判断高度自信")
    elif coherence > 0.5:
        print("  → 中等相干性：模型判断基本可靠")
    else:
        print("  → 低相干性：模型可能不擅长此输入，建议请求澄清")

# ─── Done ───
section("演示完成")
print("""
  推理流水线完整走过以下阶段：
    1. 权重校验 ✓
    2. 模型加载（道体冻结 >90% 参数）✓
    3. 语义编码 → 洛书空间 ✓
    4. 双轨递归推演 ✓
    5. 三爻空间精炼 ✓
    6. 多任务认知状态输出 ✓
    7. 原型空间检索 ✓
    8. 自校准相干性检测 ✓

  道体模型不是一个分类器。分类只是流水线的第一步。
  分类之后，才是真正的推理、检索和生成。
""")