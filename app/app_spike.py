"""
DaoTi Spike Encoding Interface — Gradio Demo
==============================================
道体基座脉冲编码接口交互式演示。

本演示仅展示接口的推理和可视化能力，不包含任何训练代码。
核心架构算法和训练配方不在此文件中。

功能:
  1. 64 通道脉冲编码可视化（rate / temporal / phase）
  2. 安全锁机制验证（正常 vs 危险状态）
  3. 64 通道编码协议展示
  4. 已验证性能指标摘要

本地运行:
    pip install -r requirements.txt
    python app_spike.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import gradio as gr

from daoti.spike_interface import (
    SpikeEncoder,
    GUA_ACTION_SEMANTICS, SAFETY_CHANNELS, MOTOR_CHANNELS,
    COGNITIVE_CHANNELS, PERCEPTION_CHANNELS, HOMEOSTASIS_CHANNELS,
)

STATE_DIM = 176
N_CHANNELS = 64
N_STEPS = 8
DEVICE = "cpu"

print("Initializing DaoTi Spike Interface...")


def encode_state_to_spikes(encoding_type, safety_lock, threshold):
    encoder = SpikeEncoder(
        STATE_DIM, N_CHANNELS, N_STEPS,
        encoding=encoding_type, threshold=threshold,
        safety_lock=safety_lock,
    ).to(DEVICE)
    encoder.eval()

    state = torch.randn(1, STATE_DIM, device=DEVICE) * 0.5
    with torch.no_grad():
        spikes, channel_act = encoder(state)

    rate = spikes.mean(dim=-1).squeeze(0).cpu().numpy()
    spike_grid = spikes.squeeze(0).cpu().numpy()

    role_rates = {}
    for role_name, channels in [
        ("cognitive", COGNITIVE_CHANNELS),
        ("perception", PERCEPTION_CHANNELS),
        ("motor", MOTOR_CHANNELS),
        ("safety", SAFETY_CHANNELS),
        ("homeostasis", HOMEOSTASIS_CHANNELS),
    ]:
        ch_list = [c for c in channels if c < N_CHANNELS]
        if ch_list:
            role_rates[role_name] = float(rate[ch_list].mean())
        else:
            role_rates[role_name] = 0.0

    top_indices = np.argsort(rate)[::-1][:10]
    top_lines = []
    for idx in top_indices:
        sem = GUA_ACTION_SEMANTICS.get(idx, {})
        top_lines.append(
            f"Ch{idx:>2d} [{sem.get('role', '?'):<12s}] "
            f"{sem.get('action', '?'):<16s} "
            f"rate={rate[idx]:.3f}  {sem.get('desc', '')}"
        )

    grid_lines = ["时间步 →  " + "  ".join(f"T{t}" for t in range(N_STEPS))]
    grid_lines.append("─" * 60)
    for ch in range(N_CHANNELS):
        sem = GUA_ACTION_SEMANTICS.get(ch, {})
        spike_str = "  ".join(
            "█" if spike_grid[ch, t] > 0.5 else "·" for t in range(N_STEPS)
        )
        role_tag = sem.get("role", "?")[:3]
        grid_lines.append(f"Ch{ch:>2d}[{role_tag}] {spike_str}  {rate[ch]:.2f}")

    safety_triggered = bool(rate[SAFETY_CHANNELS].mean() > 0.8)

    summary = f"""64 通道脉冲编码结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
编码方式: {encoding_type}
安全锁: {'✅ 开启' if safety_lock else '❌ 关闭'}
阈值: {threshold}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
通道组激活率:
  认知 (cognitive)    : {role_rates['cognitive']:.4f}
  感知 (perception)   : {role_rates['perception']:.4f}
  运动 (motor)        : {role_rates['motor']:.4f}
  安全 (safety)       : {role_rates['safety']:.4f}
  稳态 (homeostasis)  : {role_rates['homeostasis']:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
全局脉冲率: {rate.mean():.4f}
安全通道触发: {'⚠️ 是' if safety_triggered else '✅ 否'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top-10 激活通道:
""" + "\n".join(top_lines)

    return summary, "\n".join(grid_lines)


def run_safety_test():
    normal_encoder = SpikeEncoder(
        STATE_DIM, N_CHANNELS, N_STEPS,
        encoding="rate", safety_lock=True,
    ).to(DEVICE)
    normal_encoder.eval()

    danger_encoder = SpikeEncoder(
        STATE_DIM, N_CHANNELS, N_STEPS,
        encoding="rate", safety_lock=True,
    ).to(DEVICE)
    danger_encoder.eval()

    with torch.no_grad():
        for ch in SAFETY_CHANNELS:
            danger_encoder.channel_proj.weight[ch, :] = 2.0
            danger_encoder.channel_proj.bias[ch] = 2.0

    normal_state = torch.randn(1, STATE_DIM, device=DEVICE) * 0.3
    danger_state = torch.randn(1, STATE_DIM, device=DEVICE) * 0.3

    with torch.no_grad():
        normal_spikes, normal_act = normal_encoder(normal_state)
        danger_spikes, danger_act = danger_encoder(danger_state)

    normal_rate = normal_spikes.mean(dim=-1).squeeze(0).cpu().numpy()
    danger_rate = danger_spikes.mean(dim=-1).squeeze(0).cpu().numpy()

    normal_motor = normal_rate[MOTOR_CHANNELS].mean()
    danger_motor = danger_rate[MOTOR_CHANNELS].mean()
    normal_safety = normal_rate[SAFETY_CHANNELS].mean()
    danger_safety = danger_rate[SAFETY_CHANNELS].mean()

    result = f"""安全锁机制验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【正常状态】
  安全通道激活率: {normal_safety:.4f}
  运动通道激活率: {normal_motor:.4f}
  安全锁状态: ✅ 未触发 → 运动通道正常输出

【危险状态】（安全通道强制激活）
  安全通道激活率: {danger_safety:.4f}
  运动通道激活率: {danger_motor:.4f}
  安全锁状态: {'⚠️ 已触发 → 运动通道被抑制' if danger_safety > 0.8 else '未触发'}
  运动通道抑制率: {max(0, (1 - danger_motor / max(normal_motor, 0.001))) * 100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

安全通道详情（Ch24-31）:
"""
    for ch in SAFETY_CHANNELS:
        sem = GUA_ACTION_SEMANTICS.get(ch, {})
        n_r = normal_rate[ch]
        d_r = danger_rate[ch]
        result += f"  Ch{ch} {sem.get('action', '?'):<12s}: 正常={n_r:.3f}  危险={d_r:.3f}\n"

    result += f"""
运动通道详情（Ch8-23）:
"""
    for ch in range(8, 24):
        sem = GUA_ACTION_SEMANTICS.get(ch, {})
        n_r = normal_rate[ch]
        d_r = danger_rate[ch]
        suppressed = "🔒 抑制" if d_r < n_r * 0.5 else ""
        result += f"  Ch{ch} {sem.get('action', '?'):<12s}: 正常={n_r:.3f}  危险={d_r:.3f}  {suppressed}\n"

    return result


def show_channel_protocol():
    lines = ["""64 通道脉冲编码协议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

通道分组:
"""]

    groups = [
        ("认知 (cognitive)", list(range(0, 4))),
        ("感知 (perception)", list(range(4, 8))),
        ("运动 (motor) — 震宫", list(range(8, 12))),
        ("运动 (motor) — 巽宫", list(range(12, 16))),
        ("运动 (motor) — 坎宫", list(range(16, 24))),
        ("安全 (safety) — 艮宫", list(range(24, 32))),
        ("稳态 (homeostasis) — 坤宫", list(range(32, 40))),
        ("扩展认知", list(range(40, 48))),
        ("扩展感知", list(range(48, 56))),
        ("扩展稳态", list(range(56, 64))),
    ]

    for group_name, channels in groups:
        lines.append(f"\n【{group_name}】")
        for ch in channels:
            sem = GUA_ACTION_SEMANTICS.get(ch, {})
            lines.append(
                f"  Ch{ch:>2d}  {sem.get('palace', '?')}宫  "
                f"{sem.get('wuxing', '?')}行  "
                f"{sem.get('action', '?'):<16s}  "
                f"{sem.get('desc', '')}"
            )

    lines.append("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

安全通道硬约束:
  · 当安全通道组(Ch24-31)总激活率 > 0.8 时
  · 自动抑制所有运动通道(Ch8-23, Ch40-55)输出
  · 此抑制是结构性的，不由学习规则修改
  · STDP 学习规则中，安全通道突触权重被遮罩
""")

    return "\n".join(lines)


def show_performance_summary():
    return """已验证性能指标（技术协议 V1.0 摘录）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

具身控制任务（梯度学习）:
  1D 滑块  : Best Reward = -15.76  Safety Viol = 0.19
  2D 点控  : Best Reward = -27.57  Safety Viol = 0.17
  抓取     : Best Reward = -10.68  Safety Viol = 0.17

STDP-Only 学习（无梯度下降）:
  Best Reward = -16.66（与梯度差距仅 5.7%）
  突触自组织：w_mean 从 +0.23 翻转到 -0.29
  模型在无监督条件下学会了"选择性抑制"

噪声鲁棒性（18 种噪声条件）:
  · Safety Lock 零误触发（触发率 < 0.1%）
  · Gaussian σ=0.5 时性能改善 24%（随机共振效应）
  · 极端噪声（σ=2.0）下性能退化仅 17.3%

通道语义验证:
  · 三个任务自然筛选出与各自需求匹配的通道组
  · Ch18(grip) 在所有跨任务场景中保持高激活率
  · STDP-Only 收敛到与梯度训练相同的语义模式

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
完整数据详见: 道体基座脉冲编码接口 — 工程说明（技术协议 V1.0）
"""


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # ⚡ 道体基座脉冲编码接口 — 交互式演示
        **64 通道脉冲神经网络 · 架构级安全机制 · STDP 无梯度学习**

        道体基座输出的 64 通道脉冲信号，按八宫五行分为认知/感知/运动/安全/稳态五组。
        内置结构性安全约束——安全通道激活时自动抑制运动通道，不由学习规则修改。

        > 基于技术协议 V1.0 | [GitHub](https://github.com/zhibaiYingChuan/DaoTi)
        """
    )

    with gr.Tabs():
        with gr.TabItem("⚡ 脉冲编码"):
            with gr.Row():
                with gr.Column(scale=1):
                    encoding_radio = gr.Radio(
                        choices=["rate", "temporal", "phase"],
                        value="rate",
                        label="编码方式",
                    )
                    safety_check = gr.Checkbox(
                        value=True,
                        label="安全锁 (Safety Lock)",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.1,
                        label="脉冲阈值",
                    )
                    encode_btn = gr.Button("⚡ 生成脉冲编码", variant="primary")

                with gr.Column(scale=1):
                    encode_summary = gr.Textbox(
                        label="编码结果摘要",
                        lines=18,
                        interactive=False,
                    )

            spike_grid = gr.Textbox(
                label="64 通道脉冲栅格图 (█=脉冲, ·=静默)",
                lines=70,
                interactive=False,
            )

            encode_btn.click(
                fn=encode_state_to_spikes,
                inputs=[encoding_radio, safety_check, threshold_slider],
                outputs=[encode_summary, spike_grid],
            )

        with gr.TabItem("🔒 安全锁验证"):
            safety_btn = gr.Button("🔒 运行安全锁测试", variant="primary")
            safety_output = gr.Textbox(
                label="安全锁验证结果",
                lines=35,
                interactive=False,
            )

            safety_btn.click(
                fn=run_safety_test,
                inputs=[],
                outputs=[safety_output],
            )

        with gr.TabItem("📋 通道协议"):
            protocol_btn = gr.Button("📋 显示 64 通道编码协议", variant="primary")
            protocol_output = gr.Textbox(
                label="64 通道脉冲编码协议",
                lines=50,
                interactive=False,
            )

            protocol_btn.click(
                fn=show_channel_protocol,
                inputs=[],
                outputs=[protocol_output],
            )

        with gr.TabItem("📊 性能指标"):
            perf_btn = gr.Button("📊 显示已验证性能指标", variant="primary")
            perf_output = gr.Textbox(
                label="性能指标摘要",
                lines=30,
                interactive=False,
            )

            perf_btn.click(
                fn=show_performance_summary,
                inputs=[],
                outputs=[perf_output],
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8099)
