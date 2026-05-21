"""
DaoTi V53 Gradio Demo
=====================
道体基座交互式演示 — 无需编写代码，通过浏览器体验模型效果。

本地运行:
    python app.py

部署到 HuggingFace Spaces:
    1. 创建 Space (SDK: Gradio)
    2. 上传 app.py, inference.py, yijing_v53_daoti.pt, yijing_v53_config.json, requirements.txt
    3. Space 自动构建和启动
"""

import torch
import gradio as gr
from inference import (
    load_daoti, predict, generate_response, compute_coherence, verify_sha256,
    GUA_64, BA_GONG, GUA_WUXING, GUA_TRIGRAM,
    BAGUA_NAMES, WUXING_NAMES, LIUQIN_MAP, PALACE_MAP,
    sparse_expand_input, find_palace, STATE_DIM, TEXT_DIM,
)
PALACE_NAMES = ["乾宫", "坤宫", "震宫", "巽宫", "坎宫", "离宫", "艮宫", "兑宫"]
TIANGAN_NAMES = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
DIZHI_NAMES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
LIUQIN_NAMES = ["父母", "兄弟", "子孙", "妻财", "官鬼", "空亡"]
LIUSHEN_NAMES = ["青龙", "朱雀", "勾陈", "螣蛇", "白虎", "玄武"]
WANGXIANG_N = ["旺", "相", "休", "囚", "死"]
METHOD_NAMES = {"周易": "traditional", "梅花易数": "meihua", "六爻": "liuyao"}

GUA_CHOICES = [(f"{GUA_64[i]}", i) for i in range(64)]

MODEL_PATH = "yijing_v53_daoti.pt"
VOCAB_SIZE = 8145
SEQ_LEN = 256

print("Loading DaoTi V53 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_daoti(MODEL_PATH, device=device)
print(f"Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters()):,}")


def run_prediction(text_input, gua_name, method_name):
    gua_idx = GUA_64.index(gua_name)
    method = METHOD_NAMES[method_name]

    text_ids = torch.randint(1, 100, (1, SEQ_LEN), dtype=torch.long, device=device)

    result = predict(model, text_ids, gua_idx=gua_idx, method=method, device=device)

    palace_pred = PALACE_NAMES[result["palace"].argmax().item()]
    palace_conf = torch.softmax(result["palace"], dim=-1).max().item()
    liuqin_pred = LIUQIN_NAMES[result["liuqin"].argmax().item()]
    liushen_pred = LIUSHEN_NAMES[result["liushen"].argmax().item()]
    tiangan_pred = TIANGAN_NAMES[result["tiangan"].argmax().item()]
    dizhi_pred = DIZHI_NAMES[result["dizhi"].argmax().item()]
    wangxiang_pred = WANGXIANG_N[result["wangxiang"].argmax().item()]
    yao_raw = result["biangua_yao"].squeeze()
    yao_p = torch.sigmoid(yao_raw).tolist()
    yao_v = [1 if p > 0.5 else 0 for p in yao_p]
    moving_yao = [i + 1 for i, v in enumerate(yao_v) if v]
    pw_pred = WUXING_NAMES[result["palace_wuxing"].argmax().item()]
    dw_pred = WUXING_NAMES[result["dizhi_wuxing"].argmax().item()]
    coherence = result["coherence"]

    structured = f"""【{gua_name}】结构化推理结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
八宫分类  : {palace_pred}  (置信度: {palace_conf:.1%})
天干地支  : {tiangan_pred}{dizhi_pred}
六亲持世  : {liuqin_pred}
六神临爻  : {liushen_pred}
旺相休囚  : {wangxiang_pred}
宫五行    : {pw_pred}
支五行    : {dw_pred}
动爻      : {"第" + "、".join(map(str, moving_yao)) + "爻" if moving_yao else "无（静卦）"}
相干性    : {coherence:.4f}  {"⚠️ 低置信度" if coherence < 0.3 else "✅"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    rag_result = generate_response(model, text_ids, gua_idx=gua_idx, method=method, device=device)
    rag_response = rag_result["response"]

    yao_detail = "  ".join(
        [f"第{i+1}爻: {'▅▅▅ (阳)' if yao_v[i]==0 else '▅ ▅ (动)'} p={yao_p[i]:.3f}"
         for i in range(6)]
    )

    return structured, rag_response, yao_detail


def run_coherence_map(method_name):
    method = METHOD_NAMES[method_name]
    text_ids = torch.randint(1, 100, (1, SEQ_LEN), dtype=torch.long, device=device)

    results = []
    for gi in range(64):
        c = compute_coherence(model, text_ids, gi, device)
        results.append((GUA_64[gi], c))

    results.sort(key=lambda x: x[1], reverse=True)

    lines = ["64卦相干性排名（从高到低）", "━" * 40]
    for i, (name, c) in enumerate(results):
        bar = "█" * int(c * 20)
        flag = " ⚠️" if c < 0.3 else ""
        lines.append(f"{i+1:2d}. {name:<4s} {c:.4f} {bar}{flag}")

    coherences = [c for _, c in results]
    lines.append("━" * 40)
    lines.append(f"均值: {sum(coherences)/len(coherences):.4f}  "
                 f"中位数: {coherences[32]:.4f}  "
                 f"低置信度(<0.3): {sum(1 for c in coherences if c < 0.3)}/64")

    return "\n".join(lines)


def run_all_gua_stats(method_name):
    method = METHOD_NAMES[method_name]
    text_ids = torch.randint(1, 100, (1, SEQ_LEN), dtype=torch.long, device=device)

    correct = 0
    details = []
    for gi in range(64):
        r = predict(model, text_ids, gua_idx=gi, method=method, device=device)
        p_name = PALACE_NAMES[r["palace"].argmax().item()]
        gt = find_palace(GUA_64[gi])
        ok = p_name == gt
        if ok:
            correct += 1
        details.append(f"{GUA_64[gi]:<4s} → {p_name} {'✅' if ok else '❌(应为' + gt + ')'}")

    lines = [
        f"八宫分类准确率: {correct}/64 ({100*correct/64:.1f}%)",
        "━" * 40,
    ]
    for i in range(0, 64, 8):
        lines.append("  ".join(details[i:i+8]))

    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🏛️ 道体基座 DaoTi V53 — 交互式演示
        **算力不是门槛。结构性增效，而非规模堆积。**

        道体基座是一个预训练的神经网络语义基座模型，基于双轨阶梯网络架构，在消费级CPU上完成训练。
        输入中文自然语言，输出结构化语义表征（八宫、六亲、六神、天干地支、旺相休囚等）+ RAG检索增强生成。

        > ⚠️ 在线演示使用随机 token ids 模拟文本输入（分词器为受保护资产）。符号推理的计算是真实、完整、可验证的。
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔮 推理演示"):
            with gr.Row():
                with gr.Column(scale=1):
                    gua_dropdown = gr.Dropdown(
                        choices=[name for name, _ in GUA_CHOICES],
                        value="乾",
                        label="卦象选择",
                    )
                    method_radio = gr.Radio(
                        choices=list(METHOD_NAMES.keys()),
                        value="周易",
                        label="占法",
                    )
                    text_input = gr.Textbox(
                        label="输入文本（概念展示，实际使用需接入分词器）",
                        placeholder="例：今日出行是否顺利",
                        value="今日出行是否顺利",
                    )
                    predict_btn = gr.Button("🚀 开始推理", variant="primary")

                with gr.Column(scale=1):
                    structured_output = gr.Textbox(
                        label="结构化推理结果",
                        lines=14,
                        interactive=False,
                    )

            with gr.Row():
                rag_output = gr.Textbox(
                    label="RAG 检索增强生成",
                    lines=12,
                    interactive=False,
                )

            with gr.Row():
                yao_output = gr.Textbox(
                    label="爻位详情",
                    lines=2,
                    interactive=False,
                )

            predict_btn.click(
                fn=run_prediction,
                inputs=[text_input, gua_dropdown, method_radio],
                outputs=[structured_output, rag_output, yao_output],
            )

        with gr.TabItem("📊 相干性地图"):
            with gr.Row():
                coh_method = gr.Radio(
                    choices=list(METHOD_NAMES.keys()),
                    value="周易",
                    label="占法",
                )
                coh_btn = gr.Button("📊 生成相干性地图", variant="primary")

            coh_output = gr.Textbox(
                label="64卦相干性排名",
                lines=25,
                interactive=False,
            )

            coh_btn.click(
                fn=run_coherence_map,
                inputs=[coh_method],
                outputs=[coh_output],
            )

        with gr.TabItem("📈 全卦统计"):
            with gr.Row():
                stat_method = gr.Radio(
                    choices=list(METHOD_NAMES.keys()),
                    value="周易",
                    label="占法",
                )
                stat_btn = gr.Button("📈 运行全64卦统计", variant="primary")

            stat_output = gr.Textbox(
                label="八宫分类统计",
                lines=20,
                interactive=False,
            )

            stat_btn.click(
                fn=run_all_gua_stats,
                inputs=[stat_method],
                outputs=[stat_output],
            )

        with gr.TabItem("📖 关于道体"):
            gr.Markdown(
                """
                ## 道体基座核心架构

                ```
                用户输入 → 语义编码 → 领域判定 → 激活对应适配器 → 结构化推理 → 认知状态输出 → 生成回答
                ```

                ### 推理链路

                | 层级 | 模块 | 说明 |
                |:---|:---|:---|
                | 1-2 | 权重校验 + 模型加载 | >90% 参数冻结 |
                | 3 | 符号嵌入 + 文本 token | 176维洛书空间符号向量 |
                | 4 | 语义编码器 | 字符级编码（冻结） |
                | 5 | 投影层 | 语义空间映射（冻结） |
                | 6 | 门控融合 | 符号-文本自适应门控 |
                | 7 | 递归推演引擎 | 多层多步递归推理（冻结） |
                | 8 | 多任务输出头 | 共享特征+方法适配+规则推理+8任务输出 |
                | 9 | 原型空间检索 | 规范不变性验证 |
                | 10 | 方法切换 | 差异化推理 |
                | 11 | 全64卦统计 | 可复现验证 |
                | 12 | RAG 检索增强生成 | 结构化推理→自然语言 |
                | 13 | 相干性自校准 | 不确定性估计 |

                ### 关键数据

                - **参数量**: 5,059,040（约500万）
                - **训练硬件**: 消费级CPU
                - **推理延迟**: 100-200ms（CPU端到端）
                - **冻结参数**: >90%（道体，永不修改）

                ⚠️ 核心架构算法和训练配方不在此仓库中。详见 LICENSE。

                ### 许可证

                代码依据 **Apache 2.0** 许可证发布。模型权重依据 **DaoTi Research License v1.0** 发布。

                详见 [GitHub](https://github.com/zhibaiYingChuan/DaoTi)
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8088)
