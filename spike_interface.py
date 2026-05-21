"""
DaoTi Spike Encoding Interface — Public Release
================================================
道体基座脉冲编码接口（公开发布版）。

本文件定义道体基座输出的 64 通道脉冲信号的编码协议、安全约束和推理接口。
仅包含推理所需代码，不包含训练配方。

技术协议版本: V1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

BAGUA_ORDER = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
BAGUA_WUXING = {"乾": "金", "兑": "金", "坤": "土", "艮": "土",
                "震": "木", "巽": "木", "坎": "水", "离": "火"}

GUA_ACTION_SEMANTICS = {
    0: {"palace": "乾", "wuxing": "金", "role": "cognitive",
        "action": "initiate", "desc": "天行健·主动发起"},
    1: {"palace": "乾", "wuxing": "金", "role": "cognitive",
        "action": "decide", "desc": "决断·选择方向"},
    2: {"palace": "兑", "wuxing": "金", "role": "cognitive",
        "action": "communicate", "desc": "兑说·信息输出"},
    3: {"palace": "兑", "wuxing": "金", "role": "cognitive",
        "action": "exchange", "desc": "交换·双向交互"},
    4: {"palace": "离", "wuxing": "火", "role": "perception",
        "action": "illuminate", "desc": "离明·视觉感知"},
    5: {"palace": "离", "wuxing": "火", "role": "perception",
        "action": "discriminate", "desc": "辨别·模式识别"},
    6: {"palace": "离", "wuxing": "火", "role": "perception",
        "action": "detect", "desc": "检测·异常发现"},
    7: {"palace": "离", "wuxing": "火", "role": "perception",
        "action": "track", "desc": "追踪·目标锁定"},
    8: {"palace": "震", "wuxing": "木", "role": "motor",
        "action": "strike", "desc": "震动·快速打击"},
    9: {"palace": "震", "wuxing": "木", "role": "motor",
        "action": "accelerate", "desc": "加速·动能输出"},
    10: {"palace": "震", "wuxing": "木", "role": "motor",
        "action": "jump", "desc": "跃起·垂直运动"},
    11: {"palace": "震", "wuxing": "木", "role": "motor",
        "action": "vibrate", "desc": "振动·微调控制"},
    12: {"palace": "巽", "wuxing": "木", "role": "motor",
        "action": "rotate_cw", "desc": "巽入·顺时针旋转"},
    13: {"palace": "巽", "wuxing": "木", "role": "motor",
        "action": "rotate_ccw", "desc": "逆时针旋转"},
    14: {"palace": "巽", "wuxing": "木", "role": "motor",
        "action": "approach", "desc": "趋近·目标靠近"},
    15: {"palace": "巽", "wuxing": "木", "role": "motor",
        "action": "retreat", "desc": "退避·安全撤离"},
    16: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "flow_forward", "desc": "坎流·前进"},
    17: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "flow_backward", "desc": "回流·后退"},
    18: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "grip", "desc": "握持·抓取闭合"},
    19: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "release", "desc": "释放·抓取松开"},
    20: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "absorb", "desc": "吸收·能量采集"},
    21: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "discharge", "desc": "释放·能量输出"},
    22: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "cool", "desc": "冷却·降温控制"},
    23: {"palace": "坎", "wuxing": "水", "role": "motor",
        "action": "lubricate", "desc": "润滑·减阻控制"},
    24: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "stop", "desc": "艮止·紧急制动"},
    25: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "hold", "desc": "保持·位置锁定"},
    26: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "shield", "desc": "防护·安全屏障"},
    27: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "isolate", "desc": "隔离·故障隔离"},
    28: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "reset", "desc": "复位·系统重置"},
    29: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "calibrate", "desc": "校准·基准恢复"},
    30: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "idle", "desc": "待机·低功耗模式"},
    31: {"palace": "艮", "wuxing": "土", "role": "safety",
        "action": "watchdog", "desc": "看门狗·超时保护"},
    32: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "nurture", "desc": "坤养·维护修复"},
    33: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "store", "desc": "储藏·能量储备"},
    34: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "distribute", "desc": "分配·资源调度"},
    35: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "recycle", "desc": "循环·废物回收"},
    36: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "balance", "desc": "平衡·稳态调节"},
    37: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "adapt", "desc": "适应·环境适配"},
    38: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "integrate", "desc": "整合·多模态融合"},
    39: {"palace": "坤", "wuxing": "土", "role": "homeostasis",
        "action": "settle", "desc": "安顿·状态收敛"},
}

for i in range(40, 64):
    palace_idx = i // 8
    gua_name = BAGUA_ORDER[palace_idx]
    wx = BAGUA_WUXING[gua_name]
    inner_idx = i % 8
    if palace_idx == 0:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "cognitive", "action": f"think_{inner_idx}",
            "desc": f"乾宫变卦{inner_idx}·深层认知"}
    elif palace_idx == 1:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "cognitive", "action": f"comm_{inner_idx}",
            "desc": f"兑宫变卦{inner_idx}·通信协议"}
    elif palace_idx == 2:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "perception", "action": f"sense_{inner_idx}",
            "desc": f"离宫变卦{inner_idx}·感知扩展"}
    elif palace_idx == 3:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "motor", "action": f"move_{inner_idx}",
            "desc": f"震宫变卦{inner_idx}·运动变体"}
    elif palace_idx == 4:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "motor", "action": f"steer_{inner_idx}",
            "desc": f"巽宫变卦{inner_idx}·操控变体"}
    elif palace_idx == 5:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "motor", "action": f"manip_{inner_idx}",
            "desc": f"坎宫变卦{inner_idx}·操作变体"}
    elif palace_idx == 6:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "safety", "action": f"guard_{inner_idx}",
            "desc": f"艮宫变卦{inner_idx}·安全变体"}
    elif palace_idx == 7:
        GUA_ACTION_SEMANTICS[i] = {"palace": gua_name, "wuxing": wx,
            "role": "homeostasis", "action": f"maintain_{inner_idx}",
            "desc": f"坤宫变卦{inner_idx}·维护变体"}

SAFETY_CHANNELS = [24, 25, 26, 27, 28, 29, 30, 31]
MOTOR_CHANNELS = list(range(8, 24)) + list(range(40, 56))
COGNITIVE_CHANNELS = list(range(0, 8)) + list(range(40, 48))
PERCEPTION_CHANNELS = list(range(4, 8)) + list(range(48, 56))
HOMEOSTASIS_CHANNELS = list(range(32, 40)) + list(range(56, 64))

GUA_AFFINITY_SCALE = 0.3
TEMPORAL_MOD_INIT = 0.1
MEMBRANE_DECAY_INIT = 0.9


class SpikeEncoder(nn.Module):
    """
    道体基座脉冲编码器。
    将 176 维状态向量编码为 64 通道 × 8 时间步的脉冲序列。

    支持三种编码方式:
      - rate: 频率编码（脉冲概率正比于通道激活值）
      - temporal: 时间编码（激活值越高，首次脉冲越早）
      - phase: 相位编码（正弦相位调制）

    安全锁: 当安全通道组激活率 > 0.8 时，自动抑制运动通道输出。
    """
    def __init__(self, state_dim=176, n_channels=64, n_steps=8,
                 encoding="rate", threshold=0.5, safety_lock=True):
        super().__init__()
        self.state_dim = state_dim
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.encoding = encoding
        self.threshold = threshold
        self.safety_lock = safety_lock

        self.channel_proj = nn.Linear(state_dim, n_channels)
        self.temporal_modulation = nn.Parameter(torch.ones(n_channels, n_steps) * TEMPORAL_MOD_INIT)
        self.membrane_decay = nn.Parameter(torch.tensor(MEMBRANE_DECAY_INIT))

        self.register_buffer('safety_mask',
            torch.zeros(n_channels, dtype=torch.float32))
        if safety_lock:
            for ch in SAFETY_CHANNELS:
                self.safety_mask[ch] = 1.0

        self.register_buffer('refractory_counter',
            torch.zeros(1, n_channels, dtype=torch.long))

    def forward(self, state, gua_prototypes=None):
        B = state.shape[0]
        device = state.device

        channel_activation = torch.sigmoid(self.channel_proj(state))

        if gua_prototypes is not None:
            gua_affinity = torch.matmul(state, gua_prototypes.T)
            gua_weights = F.softmax(gua_affinity, dim=-1)
            channel_activation = channel_activation * (1 + GUA_AFFINITY_SCALE * gua_weights)

        if self.encoding == "rate":
            spike_probs = channel_activation.unsqueeze(2).expand(
                B, self.n_channels, self.n_steps)
            temporal_mod = torch.sigmoid(self.temporal_modulation)
            spike_probs = spike_probs * temporal_mod.unsqueeze(0)
            spike_probs = spike_probs.clamp(0, 1)
            spikes = (torch.rand_like(spike_probs) < spike_probs).float()

        elif self.encoding == "temporal":
            channel_activation = channel_activation.unsqueeze(2).expand(
                B, self.n_channels, self.n_steps)
            fire_step = (1 - channel_activation) * self.n_steps
            fire_step = fire_step.clamp(0, self.n_steps - 1).long()
            step_indices = torch.arange(self.n_steps, device=device).view(1, 1, self.n_steps)
            spikes = (step_indices >= fire_step).float()
            spikes = spikes * channel_activation

        elif self.encoding == "phase":
            channel_activation = channel_activation.unsqueeze(2)
            phase = torch.sigmoid(self.temporal_modulation).unsqueeze(0)
            t = torch.linspace(0, 2 * math.pi, self.n_steps, device=device)
            phase_signal = torch.sin(t.unsqueeze(0).unsqueeze(0) + phase * 2 * math.pi)
            spike_signal = channel_activation * phase_signal
            spikes = (spike_signal > self.threshold).float()
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        if self.safety_lock:
            safety_input = channel_activation[:, SAFETY_CHANNELS]
            safety_trigger = (safety_input > 0.8).float()
            if safety_trigger.any():
                motor_mask = torch.ones(B, self.n_channels, device=device)
                for b in range(B):
                    if safety_trigger[b].any():
                        motor_mask[b, MOTOR_CHANNELS] = 0.0
                spikes = spikes * motor_mask.unsqueeze(2)

        return spikes, channel_activation

    def decode_rate(self, spikes):
        return spikes.mean(dim=-1)

    def decode_first_spike(self, spikes):
        B, C, T = spikes.shape
        first_spike = torch.full((B, C), float(T), device=spikes.device)
        for t in range(T):
            fired = (spikes[:, :, t] > 0) & (first_spike == T)
            first_spike[fired] = t.float()
        latency = 1.0 - first_spike / T
        return latency


class SpikeDecoder(nn.Module):
    """
    脉冲解码器。
    将 64 通道脉冲序列解码为连续动作向量。

    支持三种解码模式:
      - rate: 基于脉冲频率的解码
      - temporal: 基于时间模式的解码
      - hybrid: 自适应混合解码

    安全监控: 安全通道脉冲率过高时自动缩放动作幅度。
    """
    def __init__(self, n_channels=64, n_steps=8, action_dim=8):
        super().__init__()
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.action_dim = action_dim

        self.rate_decoder = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        self.temporal_decoder = nn.Sequential(
            nn.Linear(n_channels * n_steps, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self.sync_detector = nn.Sequential(
            nn.Linear(n_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.register_buffer('safety_monitor',
            torch.zeros(n_channels, dtype=torch.float32))
        for ch in SAFETY_CHANNELS:
            self.safety_monitor[ch] = 1.0

    def forward(self, spikes, mode="rate"):
        if mode == "rate":
            rate = spikes.mean(dim=-1)
            action = self.rate_decoder(rate)
        elif mode == "temporal":
            flat = spikes.reshape(spikes.shape[0], -1)
            action = self.temporal_decoder(flat)
        elif mode == "hybrid":
            rate = spikes.mean(dim=-1)
            action_rate = self.rate_decoder(rate)
            flat = spikes.reshape(spikes.shape[0], -1)
            action_temporal = self.temporal_decoder(flat)
            sync = self.sync_detector(rate)
            action = sync * action_rate + (1 - sync) * action_temporal
        else:
            raise ValueError(f"Unknown decode mode: {mode}")

        safety_rate = spikes[:, SAFETY_CHANNELS, :].mean(dim=-1)
        safety_active = (safety_rate > 0.5).float()
        if safety_active.any():
            action_scale = 1.0 - 0.8 * safety_active.mean(dim=-1, keepdim=True)
            action = action * action_scale

        return action

    def detect_synchrony(self, spikes):
        B, C, T = spikes.shape
        motor_spikes = spikes[:, MOTOR_CHANNELS, :]
        rate = motor_spikes.mean(dim=-1)
        mean_rate = rate.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        variance = ((rate - mean_rate) ** 2).mean(dim=-1)
        synchrony_index = variance / (mean_rate ** 2 + 1e-6)
        return synchrony_index


class SpikeInterface(nn.Module):
    """
    道体基座脉冲编码接口（公开发布版）。
    输入观测向量，输出脉冲序列和动作。仅包含推理接口。

    Args:
        state_dim: 状态维度 (176)
        n_channels: 脉冲通道数 (64)
        n_steps: 时间步数 (8)
        action_dim: 动作维度 (8)
        encoding: 编码方式 ("rate" / "temporal" / "phase")
        safety_lock: 是否启用安全锁
    """
    def __init__(self, state_dim=176, n_channels=64, n_steps=8,
                 action_dim=8, encoding="rate", safety_lock=True):
        super().__init__()
        self.state_dim = state_dim
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.action_dim = action_dim

        self.encoder = SpikeEncoder(state_dim, n_channels, n_steps,
                                     encoding, safety_lock=safety_lock)
        self.decoder = SpikeDecoder(n_channels, n_steps, action_dim)

    def forward(self, state, gua_prototypes=None, decode_mode="rate"):
        """
        推理接口：状态 → 脉冲序列 → 动作

        Args:
            state: [B, state_dim] 状态向量
            gua_prototypes: [64, state_dim] 卦象原型（可选）
            decode_mode: 解码模式 ("rate" / "temporal" / "hybrid")

        Returns:
            spikes: [B, n_channels, n_steps] 64通道脉冲序列
            action: [B, action_dim] 连续动作向量
            channel_act: [B, n_channels] 通道激活值
        """
        spikes, channel_act = self.encoder(state, gua_prototypes)
        action = self.decoder(spikes, mode=decode_mode)
        return spikes, action, channel_act

    def get_channel_report(self, spikes):
        """
        生成通道激活报告。

        Args:
            spikes: [B, n_channels, n_steps] 脉冲序列

        Returns:
            dict: 每个通道的激活率和语义信息
        """
        rate = spikes.mean(dim=-1).squeeze(0)
        report = {}
        for i in range(self.n_channels):
            sem = GUA_ACTION_SEMANTICS.get(i, {})
            report[i] = {
                "rate": rate[i].item(),
                "role": sem.get("role", "unknown"),
                "action": sem.get("action", "unknown"),
                "desc": sem.get("desc", ""),
            }
        return report
