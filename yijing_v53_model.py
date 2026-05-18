"""
YiJing V53 Foundation Model — Research Preview / Reference Implementation
================================================================================
A neural network model for I Ching (易经) divination and analysis,
featuring a dual-pathway architecture with symbolic reasoning and text understanding.

IMPORTANT — READ THE LICENSE (LICENSE file) BEFORE USE:
  - This file (Reference Code) is publicly available under the DaoTi Research License.
  - Model Weights (safetensors, .pt, etc.) are NOT included in this repository.
  - Model Weights require separate written authorization from SmallLoong Research.
  - See LICENSE Section 2.2 for weight access procedures.

Architecture Overview:
  TextEncoder (128-dim) -> TextProj (176-dim) -> HeLuoLadderNetwork (6-layer)
      + SymbolicInput (176-dim) -> Gate Fusion -> Ladder -> Output Heads

Core Components:
  - TextEncoder: 2-layer Transformer encoder for text understanding
  - HeLuoLadderNetwork: Bidirectional ladder network with I Ching structural priors
  - TrigramSpace: Yin-Yang bifurcation + Wuxing curvature + Bagua sphere mapping
  - OutputHeadV38: Multi-method output with LoRA adaptation and Wuxing Shengke rules

Original Research:
  - Degenerate Ground State theory & Gauge Field structure
  - Fixed Point discovery (text_proj invariance)
  - Frozen DaoTi + Lightweight Adapter paradigm

License: DaoTi Research License (see LICENSE)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# ==============================================================================
# I Ching Domain Constants
# ==============================================================================

MAX_SEQ = 256
STATE_DIM = 176
TEXT_DIM = 128

GUA_64 = [
    "乾", "坤", "屯", "蒙", "需", "讼", "师", "比", "小畜", "履", "泰", "否",
    "同人", "大有", "谦", "豫", "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
    "无妄", "大畜", "颐", "大过", "坎", "离", "咸", "恒", "遁", "大壮", "晋", "明夷",
    "家人", "睽", "蹇", "解", "损", "益", "夬", "姤", "萃", "升", "困", "井",
    "革", "鼎", "震", "艮", "渐", "归妹", "丰", "旅", "巽", "兑", "涣", "节",
    "中孚", "小过", "既济", "未济",
]

BA_GONG = {
    "乾宫": ["乾", "姤", "遁", "否", "观", "剥", "晋", "大有"],
    "坤宫": ["坤", "复", "临", "泰", "大壮", "夬", "需", "比"],
    "震宫": ["震", "豫", "解", "恒", "升", "井", "大过", "随"],
    "巽宫": ["巽", "小畜", "家人", "益", "无妄", "噬嗑", "颐", "蛊"],
    "坎宫": ["坎", "节", "屯", "既济", "革", "丰", "明夷", "师"],
    "离宫": ["离", "旅", "鼎", "未济", "蒙", "涣", "讼", "同人"],
    "艮宫": ["艮", "贲", "大畜", "损", "睽", "履", "中孚", "渐"],
    "兑宫": ["兑", "困", "萃", "咸", "蹇", "谦", "小过", "归妹"],
}

GUA_WUXING = {
    "乾": "金", "兑": "金", "坤": "土", "艮": "土",
    "震": "木", "巽": "木", "坎": "水", "离": "火",
}

XIAN_TIAN_MAP = {
    0: 0, 1: 7, 2: 4, 3: 6, 4: 5, 5: 0, 6: 5, 7: 7,
    8: 1, 9: 1, 10: 7, 11: 0, 12: 2, 13: 0, 14: 6, 15: 4,
    16: 3, 17: 5, 18: 7, 19: 5, 20: 2, 21: 2, 22: 6, 23: 4,
    24: 0, 25: 6, 26: 3, 27: 1, 28: 5, 29: 2, 30: 6, 31: 3,
    32: 5, 33: 0, 34: 2, 35: 2, 36: 5, 37: 1, 38: 8, 39: 5,
    40: 6, 41: 4, 42: 0, 43: 5, 44: 6, 45: 4, 46: 6, 47: 5,
    48: 2, 49: 5, 50: 3, 51: 6, 52: 5, 53: 1, 54: 2, 55: 8,
    56: 5, 57: 1, 58: 3, 59: 0, 60: 6, 61: 8, 62: 0, 63: 9,
}

HOU_TIAN_MAP = {
    0: 5, 1: 1, 2: 2, 3: 8, 4: 0, 5: 5, 6: 0, 7: 1,
    8: 3, 9: 6, 10: 1, 11: 5, 12: 9, 13: 5, 14: 8, 15: 2,
    16: 2, 17: 3, 18: 1, 19: 3, 20: 9, 21: 9, 22: 8, 23: 2,
    24: 5, 25: 8, 26: 2, 27: 6, 28: 0, 29: 9, 30: 6, 31: 8,
    32: 3, 33: 5, 34: 9, 35: 9, 36: 3, 37: 6, 38: 8, 39: 0,
    40: 8, 41: 3, 42: 5, 43: 3, 44: 6, 45: 3, 46: 6, 47: 3,
    48: 9, 49: 3, 50: 2, 51: 8, 52: 3, 53: 6, 54: 9, 55: 8,
    56: 3, 57: 6, 58: 3, 59: 0, 60: 6, 61: 8, 62: 0, 63: 9,
}

GUA_TRIGRAM = {
    0: ("乾", "乾"), 1: ("坤", "坤"), 2: ("坎", "震"), 3: ("艮", "坎"),
    4: ("坎", "乾"), 5: ("乾", "坎"), 6: ("坤", "坎"), 7: ("坤", "坎"),
    8: ("巽", "乾"), 9: ("乾", "兑"), 10: ("坤", "乾"), 11: ("乾", "坤"),
    12: ("乾", "离"), 13: ("离", "乾"), 14: ("艮", "坤"), 15: ("坤", "震"),
    16: ("兑", "震"), 17: ("艮", "巽"), 18: ("坤", "兑"), 19: ("坤", "巽"),
    20: ("离", "震"), 21: ("艮", "离"), 22: ("艮", "坤"), 23: ("坤", "震"),
    24: ("乾", "震"), 25: ("艮", "乾"), 26: ("艮", "震"), 27: ("兑", "巽"),
    28: ("坎", "坎"), 29: ("离", "离"), 30: ("兑", "艮"), 31: ("震", "巽"),
    32: ("巽", "乾"), 33: ("震", "乾"), 34: ("离", "坤"), 35: ("坤", "离"),
    36: ("巽", "离"), 37: ("离", "兑"), 38: ("坎", "艮"), 39: ("震", "坎"),
    40: ("艮", "兑"), 41: ("巽", "震"), 42: ("乾", "兑"), 43: ("乾", "巽"),
    44: ("兑", "坤"), 45: ("坤", "巽"), 46: ("兑", "坎"), 47: ("巽", "坎"),
    48: ("兑", "离"), 49: ("离", "巽"), 50: ("震", "震"), 51: ("艮", "艮"),
    52: ("艮", "巽"), 53: ("震", "兑"), 54: ("离", "震"), 55: ("离", "艮"),
    56: ("巽", "巽"), 57: ("兑", "兑"), 58: ("巽", "坎"), 59: ("坎", "兑"),
    60: ("巽", "兑"), 61: ("艮", "震"), 62: ("坎", "离"), 63: ("离", "坎"),
}

WUXING_IDX = {"金": 0, "木": 1, "水": 2, "火": 3, "土": 4}
LIUQIN_MAP = {"父母": 0, "兄弟": 1, "子孙": 2, "妻财": 3, "官鬼": 4, "空亡": 5}
PALACE_MAP = {"乾宫": 0, "坤宫": 1, "震宫": 2, "巽宫": 3, "坎宫": 4, "离宫": 5, "艮宫": 6, "兑宫": 7}
METHOD_MAP = {"周易": 0, "六爻": 1, "梅花": 2}

BAGUA_NAMES = ["乾", "坤", "震", "巽", "坎", "离", "艮", "兑"]
WUXING_NAMES = ["木", "火", "土", "金", "水"]
WUXING_SHENG = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
WUXING_KE = {"木": "土", "火": "金", "土": "水", "金": "木", "水": "火"}
XIAN_TIAN_BAGUA_ORDER = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
HOU_TIAN_BAGUA_ORDER = ["离", "坤", "兑", "乾", "坎", "艮", "震", "巽"]


def _build_wuxing_shengke_matrix():
    def _get_liuqin(p, n):
        if p == n:
            return "兄弟"
        if (p, n) in [("木", "水"), ("火", "木"), ("土", "火"), ("金", "土"), ("水", "金")]:
            return "子孙"
        if (p, n) in [("木", "金"), ("火", "木"), ("土", "水"), ("金", "火"), ("水", "土")]:
            return "官鬼"
        if (p, n) in [("木", "土"), ("火", "金"), ("土", "木"), ("金", "水"), ("水", "火")]:
            return "妻财"
        return "父母"

    matrix = np.zeros((5, 5, 6), dtype=np.float32)
    for pi, pn in enumerate(["金", "木", "水", "火", "土"]):
        for ni, nn_wx in enumerate(["金", "木", "水", "火", "土"]):
            lq = _get_liuqin(pn, nn_wx)
            matrix[pi, ni, LIUQIN_MAP[lq]] = 1.0
    return matrix


WUXING_SHENGKE_MATRIX = _build_wuxing_shengke_matrix()


def gua_to_yao_bits(gi):
    return [float((gi >> j) & 1) for j in range(5, -1, -1)]


def find_palace(g):
    for p, gs in BA_GONG.items():
        if g in gs:
            return p
    return "乾宫"


def sparse_expand_input(gi, yp=0, yi=0, ri=0, si=0, mi=0):
    x = np.zeros(STATE_DIM, dtype=np.float32)
    for j in range(6):
        x[j] = float(((gi >> (5 - j)) & 1) * 2 - 1)
    xt = XIAN_TIAN_MAP.get(gi, 0)
    if xt < 8:
        x[6 + xt] = 1.0
    ht = HOU_TIAN_MAP.get(gi, 0)
    if ht < 9:
        x[14 + ht] = 1.0
    upper_gua, lower_gua = GUA_TRIGRAM.get(gi, ("乾", "乾"))
    gua_wuxing = GUA_WUXING.get(upper_gua, "金")
    wx_idx = WUXING_IDX.get(gua_wuxing, 0)
    x[23 + wx_idx] = 1.0
    palace = find_palace(GUA_64[gi])
    palace_idx = PALACE_MAP.get(palace, 0)
    x[28 + palace_idx] = 1.0
    return x


# ==============================================================================
# TextEncoder
# ==============================================================================

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=3000, embed_dim=64, hidden_dim=128,
                 num_heads=4, num_layers=2, max_seq=256, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim))

    def forward(self, text_ids, mask=None):
        B, L = text_ids.shape
        positions = torch.arange(L, device=text_ids.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(text_ids) + self.pos_embed(positions)
        h = self.layer_norm(h)
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        padding_mask = (text_ids == 0)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        non_pad = (text_ids != 0).unsqueeze(-1).float()
        h_pooled = (h * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)
        return self.pool_proj(h_pooled), h


# ==============================================================================
# MLMHead
# ==============================================================================

class MLMHead(nn.Module):
    def __init__(self, hidden_dim=64, vocab_size=3000):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size))

    def forward(self, hidden_states):
        return self.fc(hidden_states)


# ==============================================================================
# HeLuoLadderNetwork
# ==============================================================================

class HeLuoLadderCell(nn.Module):
    def __init__(self, state_dim, hidden_dim, gua_embed_dim=32, dropout=0.2):
        super().__init__()
        self.gua_embed_dim = gua_embed_dim
        self.gua_position_embed = nn.Embedding(64, gua_embed_dim)
        self.xiantian_embed = nn.Embedding(8, gua_embed_dim // 2)
        self.houtian_embed = nn.Embedding(9, gua_embed_dim // 2)
        xt_lut = torch.zeros(64, dtype=torch.long)
        ht_lut = torch.zeros(64, dtype=torch.long)
        for i in range(64):
            xt_lut[i] = min(XIAN_TIAN_MAP.get(i, 0), 7)
            ht_lut[i] = min(HOU_TIAN_MAP.get(i, 0), 8)
        self.register_buffer('xt_lut', xt_lut)
        self.register_buffer('ht_lut', ht_lut)
        combined_dim = gua_embed_dim + gua_embed_dim // 2 + gua_embed_dim // 2
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, state_dim))
        self.backward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, state_dim))
        self.forward_gate = nn.Linear(state_dim * 2 + combined_dim, state_dim)
        self.backward_gate = nn.Linear(state_dim * 2 + combined_dim, state_dim)
        self.layer_dropout = nn.Dropout(dropout * 0.5)

    def forward(self, state, gua_idx):
        gua_vec = self.gua_position_embed(gua_idx)
        xt_vec = self.xiantian_embed(self.xt_lut[gua_idx])
        ht_vec = self.houtian_embed(self.ht_lut[gua_idx])
        direction_vec = torch.cat([gua_vec, xt_vec, ht_vec], dim=-1)
        forward_pred = self.forward_net(state)
        forward_gate_val = torch.sigmoid(
            self.forward_gate(torch.cat([state, forward_pred, direction_vec], dim=-1)))
        forward_state = forward_gate_val * forward_pred + (1 - forward_gate_val) * state
        forward_state = self.layer_dropout(forward_state)
        backward_pred = self.backward_net(state)
        backward_gate_val = torch.sigmoid(
            self.backward_gate(torch.cat([state, backward_pred, direction_vec], dim=-1)))
        backward_state = backward_gate_val * backward_pred + (1 - backward_gate_val) * state
        backward_state = self.layer_dropout(backward_state)
        return forward_state, backward_state


class HeLuoLadderNetwork(nn.Module):
    def __init__(self, input_dim=196, state_dim=176, hidden_dim=320,
                 num_layers=6, T=7, dropout=0.2, gua_embed_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T = T
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim), nn.Dropout(dropout * 0.8))
        self.ladder_layers = nn.ModuleList([
            HeLuoLadderCell(state_dim, hidden_dim, gua_embed_dim=gua_embed_dim, dropout=dropout)
            for _ in range(num_layers)])
        self.output_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim), nn.Dropout(dropout * 0.8))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x, gua_idx, return_state=False):
        state = self.input_encoder(x)
        for t in range(self.T):
            new_state = torch.zeros_like(state)
            for layer in self.ladder_layers:
                forward_state, backward_state = layer(state, gua_idx)
                new_state = new_state + (forward_state + backward_state) / 2.0
            if t < self.T - 1:
                new_state = F.dropout(new_state, p=0.1, training=self.training)
            state = new_state / self.num_layers
        attn_out, _ = self.multihead_attn(
            state.unsqueeze(1), state.unsqueeze(1), state.unsqueeze(1))
        output = self.output_decoder(attn_out.squeeze(1))
        output = self.norm(output)
        if return_state:
            return output, state
        return output


# ==============================================================================
# WuxingShengkeModule
# ==============================================================================

class WuxingShengkeModule(nn.Module):
    def __init__(self, h=176, dropout=0.2):
        super().__init__()
        half_h = h // 2
        self.palace_wuxing_fc = nn.Linear(h, 5)
        self.dizhi_wuxing_fc = nn.Linear(h, 5)
        self.register_buffer('shengke_matrix', torch.from_numpy(WUXING_SHENGKE_MATRIX))
        self.liuqin_residual = nn.Sequential(
            nn.Linear(h + 6, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6))
        self.liushen_rizhi_fc = nn.Linear(h, 10)
        self.liushen_residual = nn.Sequential(
            nn.Linear(h + 6 + 10, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6))

    def forward(self, x):
        palace_wx_logits = self.palace_wuxing_fc(x)
        dizhi_wx_logits = self.dizhi_wuxing_fc(x)
        palace_wx = F.softmax(palace_wx_logits, dim=-1)
        dizhi_wx = F.softmax(dizhi_wx_logits, dim=-1)
        rule_liuqin = torch.einsum('bp,pnk,bn->bk', palace_wx, self.shengke_matrix, dizhi_wx)
        liuqin_res = self.liuqin_residual(torch.cat([x, rule_liuqin], dim=-1))
        liuqin_out = rule_liuqin + liuqin_res
        rizhi_logits = self.liushen_rizhi_fc(x)
        liushen_res = self.liushen_residual(torch.cat([x, rule_liuqin, rizhi_logits], dim=-1))
        return liuqin_out, liushen_res, palace_wx_logits, dizhi_wx_logits


# ==============================================================================
# OutputHeadV38
# ==============================================================================

class OutputHeadV38(nn.Module):
    def __init__(self, h=176, dropout=0.2, lora_rank=8, lora_alpha=0.1, enable_lora=True):
        super().__init__()
        self.h = h
        self.enable_lora = enable_lora
        self.lora_alpha = lora_alpha
        half_h = h // 2
        self.shared_fc = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h))
        self.semantic_prototypes = nn.Parameter(torch.randn(8, half_h) * 0.02)
        self.classify_q = nn.Linear(h, half_h)
        self.classify_attn = nn.MultiheadAttention(
            embed_dim=half_h, num_heads=4, dropout=dropout, batch_first=True)
        self.wuxing_shengke = WuxingShengkeModule(h, dropout)
        self.palace = nn.Linear(h, 8)
        self.tiangan = nn.Linear(h, 10)
        self.dizhi = nn.Linear(h, 12)
        self.wangxiang = nn.Linear(h + half_h, 5)
        self.biangua_yao = nn.Sequential(
            nn.Linear(h, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6))
        if enable_lora:
            self.lora_A = nn.ModuleDict({
                name: nn.Linear(h, lora_rank, bias=False)
                for name in ['traditional', 'meihua', 'liuyao']})
            self.lora_B = nn.ModuleDict({
                name: nn.Linear(lora_rank, h, bias=False)
                for name in ['traditional', 'meihua', 'liuyao']})
            for m in ['traditional', 'meihua', 'liuyao']:
                nn.init.normal_(self.lora_A[m].weight, std=0.02)
                nn.init.zeros_(self.lora_B[m].weight)

    def forward(self, x, method_name='traditional'):
        half_h = self.h // 2
        shared_features = self.shared_fc(x)
        if self.enable_lora and method_name in self.lora_A:
            lora_delta = self.lora_B[method_name](self.lora_A[method_name](shared_features))
            shared_features = shared_features + self.lora_alpha * lora_delta
        q = self.classify_q(shared_features).unsqueeze(1)
        prototypes = self.semantic_prototypes.unsqueeze(0).expand(q.size(0), -1, -1)
        classified, _ = self.classify_attn(q, prototypes, prototypes)
        classified = classified.squeeze(1)
        liuqin_out, liushen_out, palace_wx_logits, dizhi_wx_logits = self.wuxing_shengke(x)
        standard_out = torch.cat([shared_features, classified], dim=-1)
        return {
            'palace': self.palace(shared_features),
            'tiangan': self.tiangan(shared_features),
            'dizhi': self.dizhi(shared_features),
            'liuqin': liuqin_out,
            'liushen': liushen_out,
            'wangxiang': self.wangxiang(standard_out),
            'biangua_yao': self.biangua_yao(shared_features),
            'palace_wuxing': palace_wx_logits,
            'dizhi_wuxing': dizhi_wx_logits,
        }


# ==============================================================================
# TrigramSpace (DaoTi Core)
# ==============================================================================

class YinYangBifurcator(nn.Module):
    def __init__(self, input_dim=128, yin_dim=64, yang_dim=64):
        super().__init__()
        self.yang_proj = nn.Sequential(
            nn.Linear(input_dim, yang_dim), nn.GELU(), nn.LayerNorm(yang_dim))
        self.yin_proj = nn.Sequential(
            nn.Linear(input_dim, yin_dim), nn.GELU(), nn.LayerNorm(yin_dim))
        self.gate = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, x):
        yang = self.yang_proj(x)
        yin = self.yin_proj(x)
        alpha = self.gate(x)
        yang_out = yang * alpha
        yin_out = yin * (1.0 - alpha)
        return yang_out, yin_out, alpha.squeeze(-1)


class WuxingCurvatureGenerator(nn.Module):
    def __init__(self, state_dim=176, n_heads=5):
        super().__init__()
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.head_dim = 36
        self.padded_dim = self.head_dim * n_heads
        self.wuxing_query = nn.Linear(state_dim, self.padded_dim, bias=False)
        self.wuxing_key = nn.Linear(state_dim, self.padded_dim, bias=False)
        self.wuxing_value = nn.Linear(state_dim, self.padded_dim, bias=False)
        self.out_proj = nn.Linear(self.padded_dim, state_dim)
        sheng_matrix = torch.zeros(n_heads, n_heads)
        ke_matrix = torch.zeros(n_heads, n_heads)
        for i in range(n_heads):
            sheng_matrix[i, (i + 1) % n_heads] = 1.0
            ke_matrix[i, (i + 2) % n_heads] = 1.0
        self.register_buffer("sheng_mask", sheng_matrix)
        self.register_buffer("ke_mask", ke_matrix)
        self.sheng_scale = nn.Parameter(torch.tensor(2.0))
        self.ke_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        B = x.shape[0]
        Q = self.wuxing_query(x).view(B, self.n_heads, self.head_dim)
        K = self.wuxing_key(x).view(B, self.n_heads, self.head_dim)
        V = self.wuxing_value(x).view(B, self.n_heads, self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        sheng_bias = self.sheng_mask.unsqueeze(0).expand(B, -1, -1) * self.sheng_scale
        ke_bias = self.ke_mask.unsqueeze(0).expand(B, -1, -1) * self.ke_scale
        attn = attn + sheng_bias + ke_bias
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.contiguous().view(B, self.padded_dim)
        return self.out_proj(out)


class BaguaSphereMapper(nn.Module):
    def __init__(self, state_dim=176, sphere_dim=3):
        super().__init__()
        self.state_dim = state_dim
        self.sphere_dim = sphere_dim
        self.to_sphere = nn.Sequential(
            nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, sphere_dim))
        theta_positions = torch.tensor(
            [2 * math.pi * i / 8 for i in range(8)], dtype=torch.float32)
        phi_positions = torch.tensor(
            [math.pi * (0.25 + 0.5 * i / 7) for i in range(8)], dtype=torch.float32)
        xiantian_basis = torch.zeros(8, sphere_dim)
        for i in range(8):
            theta, phi = theta_positions[i], phi_positions[i]
            xiantian_basis[i, 0] = math.sin(phi) * math.cos(theta)
            xiantian_basis[i, 1] = math.sin(phi) * math.sin(theta)
            xiantian_basis[i, 2] = math.cos(phi)
        self.register_buffer("xiantian_basis", xiantian_basis)
        houtian_offset = torch.tensor(
            [2 * math.pi * i / 8 + math.pi / 8 for i in range(8)], dtype=torch.float32)
        houtian_basis = torch.zeros(8, sphere_dim)
        for i in range(8):
            theta, phi = houtian_offset[i], phi_positions[7 - i]
            houtian_basis[i, 0] = math.sin(phi) * math.cos(theta)
            houtian_basis[i, 1] = math.sin(phi) * math.sin(theta)
            houtian_basis[i, 2] = math.cos(phi)
        self.register_buffer("houtian_basis", houtian_basis)
        self.flow_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        sphere_coord = self.to_sphere(x)
        sphere_coord = F.normalize(sphere_coord, dim=-1)
        xiantian_sim = F.cosine_similarity(
            sphere_coord.unsqueeze(1), self.xiantian_basis.unsqueeze(0), dim=-1)
        houtian_sim = F.cosine_similarity(
            sphere_coord.unsqueeze(1), self.houtian_basis.unsqueeze(0), dim=-1)
        alpha = torch.sigmoid(self.flow_weight)
        combined_sim = alpha * xiantian_sim + (1 - alpha) * houtian_sim
        return {
            "sphere_coord": sphere_coord,
            "xiantian_sim": xiantian_sim,
            "houtian_sim": houtian_sim,
            "combined_sim": combined_sim,
            "flow_weight": alpha,
        }


class HeluoInteractionFolder(nn.Module):
    def __init__(self, state_dim=176, n_gua=64, bottleneck_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.n_gua = n_gua
        self.bottleneck_dim = bottleneck_dim
        self.query_proj = nn.Linear(state_dim, state_dim)
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(state_dim)
        self.sheng_scale = nn.Parameter(torch.tensor(1.0))
        self.ke_scale_raw = nn.Parameter(torch.tensor(0.5))
        gua_wuxing_idx = torch.zeros(n_gua, dtype=torch.long)
        bagua_order = ["乾", "坤", "震", "巽", "坎", "离", "艮", "兑"]
        wuxing_map = {"金": 0, "木": 1, "水": 2, "火": 3, "土": 4}
        for palace_idx, gua_name in enumerate(bagua_order):
            wx = GUA_WUXING[gua_name]
            wx_idx = wuxing_map[wx]
            for j in range(8):
                gua_idx = palace_idx * 8 + j
                if gua_idx < n_gua:
                    gua_wuxing_idx[gua_idx] = wx_idx
        self.register_buffer("gua_wuxing_idx", gua_wuxing_idx)
        wuxing_sheng = {0: 2, 1: 3, 2: 0, 3: 4, 4: 1}
        wuxing_ke = {0: 1, 1: 4, 2: 3, 3: 0, 4: 2}
        n_wuxing = 5
        wuxing_sheng_bias = torch.zeros(n_wuxing, n_gua)
        wuxing_ke_bias = torch.zeros(n_wuxing, n_gua)
        for wx_i in range(n_wuxing):
            for j in range(n_gua):
                wx_j = gua_wuxing_idx[j].item()
                if wx_i == wx_j:
                    wuxing_sheng_bias[wx_i, j] = 0.3
                elif wuxing_sheng.get(wx_i, -1) == wx_j:
                    wuxing_sheng_bias[wx_i, j] = 1.0
                elif wuxing_ke.get(wx_i, -1) == wx_j:
                    wuxing_ke_bias[wx_i, j] = 1.0
        self.register_buffer("wuxing_sheng_bias", wuxing_sheng_bias)
        self.register_buffer("wuxing_ke_bias", wuxing_ke_bias)
        self.coherence_gate_net = nn.Sequential(
            nn.Linear(state_dim + 1, state_dim // 4), nn.GELU(),
            nn.Linear(state_dim // 4, state_dim))
        self.coherence_gate_scale = nn.Parameter(torch.tensor(2.0))
        self.wave_feedback_net = nn.Sequential(
            nn.Linear(state_dim + state_dim, state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, state_dim))
        self.wave_feedback_gate = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("cached_wave_direction", torch.tensor(1.0))
        self.bottleneck_compress = nn.Linear(state_dim, bottleneck_dim, bias=False)
        self.bottleneck_expand = nn.Linear(bottleneck_dim, state_dim, bias=False)
        self.register_buffer("cached_bottleneck_ratio", torch.tensor(0.5))

    def forward(self, query_vec, proto_vecs, coherence=None, wave_feedback=None):
        B = query_vec.shape[0]
        Q = self.query_proj(query_vec)
        K = self.key_proj(proto_vecs)
        V = self.value_proj(proto_vecs)
        attn = torch.matmul(Q, K.T) * self.scale
        with torch.no_grad():
            top1_idx = attn.argmax(dim=-1)
            top1_wuxing = self.gua_wuxing_idx[top1_idx]
        sheng_bias = self.wuxing_sheng_bias[top1_wuxing] * self.sheng_scale
        ke_bias = self.wuxing_ke_bias[top1_wuxing] * (-torch.abs(self.ke_scale_raw))
        attn = attn + sheng_bias + ke_bias
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, V)
        out = self.out_proj(context)
        folded = query_vec + out
        if coherence is not None:
            coh_signal = coherence.unsqueeze(-1)
            gate_input = torch.cat([folded.detach(), coh_signal], dim=-1)
            gate_raw = self.coherence_gate_net(gate_input)
            gate_centered = gate_raw - gate_raw.mean(dim=-1, keepdim=True)
            coherence_gate = torch.sigmoid(
                gate_centered * self.coherence_gate_scale + coh_signal * 3.0)
            coherence_gate = coherence_gate.clamp(0.05, 1.0)
            folded = folded * coherence_gate
        else:
            coherence_gate = None
        if wave_feedback is not None:
            fb_input = torch.cat([folded.detach(), wave_feedback], dim=-1)
            fb_signal = self.wave_feedback_net(fb_input)
            fb_direction = self.cached_wave_direction
            fb_gate = torch.sigmoid(self.wave_feedback_gate)
            folded = folded + fb_gate * fb_signal * fb_direction
        bn_ratio = self.cached_bottleneck_ratio
        compressed = self.bottleneck_compress(folded)
        expanded = self.bottleneck_expand(compressed)
        folded = folded * (1.0 - bn_ratio) + expanded * bn_ratio
        folded = self.norm(folded)
        return {
            "folded": folded,
            "coherence_gate": coherence_gate,
            "bottleneck_ratio": bn_ratio,
        }


class DomainClassifier(nn.Module):
    def __init__(self, state_dim=176, n_domains=8, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_domains)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.norm(h)
        return self.fc2(h)


class ConstrainedPenetration(nn.Module):
    def __init__(self, dim, dropout=0.3, noise_std=0.05):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std
        self.gate = nn.Parameter(torch.ones(dim) * 0.5)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate)
        gated = x * gate
        dropped = self.dropout(gated)
        if self.training and self.noise_std > 0:
            dropped = dropped + torch.randn_like(dropped) * self.noise_std
        return self.norm(dropped)


class ResonanceCavity(nn.Module):
    def __init__(self, state_dim=176, n_domains=8, momentum=0.9,
                 coherence_mode="separation"):
        super().__init__()
        self.state_dim = state_dim
        self.n_domains = n_domains
        self.momentum = momentum
        self.coherence_mode = coherence_mode
        self.register_buffer("standing_wave", torch.zeros(n_domains, state_dim))
        self.register_buffer("wave_energy", torch.zeros(n_domains))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("ema_coh_correct", torch.tensor(0.5))
        self.register_buffer("ema_coh_wrong", torch.tensor(0.5))
        self.register_buffer("ema_gap_updates", torch.tensor(0))
        self.register_buffer("ema_wave_direction", torch.tensor(1.0))
        self.register_buffer("ema_wave_direction_updates", torch.tensor(0))
        self.register_buffer("prev_wave_energy_sum", torch.tensor(0.0))
        self.coherence_proj = nn.Linear(state_dim, state_dim // 2, bias=False)
        self.wave_proj = nn.Linear(state_dim, state_dim // 2, bias=False)
        self.modulation_net = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.mod_bias = nn.Parameter(torch.tensor(0.5))

    @torch.no_grad()
    def update_standing_wave(self, folded, domain_labels, coherence=None, predictions=None):
        for di in range(self.n_domains):
            mask = domain_labels == di
            if mask.sum() == 0:
                continue
            domain_mean = folded[mask].mean(dim=0)
            domain_energy = (folded[mask] - domain_mean).pow(2).sum(dim=-1).mean()
            if not self.initialized:
                self.standing_wave[di] = domain_mean
                self.wave_energy[di] = domain_energy
            else:
                self.standing_wave[di] = (
                    self.momentum * self.standing_wave[di] + (1 - self.momentum) * domain_mean)
                self.wave_energy[di] = (
                    self.momentum * self.wave_energy[di] + (1 - self.momentum) * domain_energy)
        if not self.initialized:
            self.initialized.fill_(True)

    def compute_coherence(self, folded, domain_logits):
        if not self.initialized:
            return torch.ones(folded.shape[0], device=folded.device) * 0.5
        if self.coherence_mode == "separation":
            return self._compute_coherence_separation(folded, domain_logits)
        elif self.coherence_mode == "combined":
            return self._compute_coherence_combined(folded, domain_logits)
        else:
            return self._compute_coherence_wave(folded, domain_logits)

    def _compute_coherence_wave(self, folded, domain_logits):
        domain_probs = F.softmax(domain_logits, dim=-1)
        wave_mix = torch.matmul(domain_probs, self.standing_wave)
        folded_proj = self.coherence_proj(folded)
        wave_proj = self.wave_proj(wave_mix)
        coherence = F.cosine_similarity(folded_proj, wave_proj, dim=-1)
        return (coherence + 1.0) / 2.0

    def _compute_coherence_separation(self, folded, domain_logits):
        if not self.initialized:
            return torch.ones(folded.shape[0], device=folded.device) * 0.5
        sw_norm = F.normalize(self.standing_wave, dim=-1)
        if sw_norm.isnan().any():
            return torch.ones(folded.shape[0], device=folded.device) * 0.5
        folded_norm = F.normalize(folded, dim=-1)
        sim_to_all = torch.matmul(folded_norm, sw_norm.T)
        domain_probs = F.softmax(domain_logits, dim=-1)
        pred_domain = domain_probs.argmax(dim=-1)
        sim_to_pred = sim_to_all.gather(1, pred_domain.unsqueeze(1)).squeeze(1)
        wrong_mask = torch.ones_like(sim_to_all, dtype=torch.bool)
        wrong_mask.scatter_(1, pred_domain.unsqueeze(1), False)
        max_wrong_sim = sim_to_all.masked_fill(~wrong_mask, -2.0).max(dim=-1).values
        separation = sim_to_pred - max_wrong_sim
        coherence = (separation + 1.0) / 2.0
        return coherence.clamp(0.0, 1.0)

    def _compute_coherence_combined(self, folded, domain_logits):
        if not self.initialized:
            return torch.ones(folded.shape[0], device=folded.device) * 0.5
        wave_coh = self._compute_coherence_wave(folded, domain_logits)
        sep_coh = self._compute_coherence_separation(folded, domain_logits)
        return torch.sqrt(wave_coh * sep_coh + 1e-8)

    def compute_wave_feedback(self, folded, domain_logits):
        if not self.initialized:
            return torch.zeros_like(folded)
        domain_probs = F.softmax(domain_logits, dim=-1)
        wave_mix = torch.matmul(domain_probs, self.standing_wave)
        energy_mix = torch.matmul(
            domain_probs, self.wave_energy.unsqueeze(-1)).squeeze(-1)
        energy_scale = torch.sigmoid(energy_mix).unsqueeze(-1)
        return wave_mix * energy_scale

    def compute_modulation(self, coherence, domain_logits):
        if not self.initialized:
            return torch.ones(coherence.shape[0], 1, device=coherence.device) * 0.5
        domain_probs = F.softmax(domain_logits, dim=-1)
        domain_conf = domain_probs.max(dim=-1).values
        wave_energy_mix = torch.matmul(
            domain_probs, self.wave_energy.unsqueeze(-1)).squeeze(-1)
        energy_norm = torch.sigmoid(wave_energy_mix)
        mod_input = torch.stack([coherence, domain_conf, energy_norm], dim=-1)
        mod_raw = self.modulation_net(mod_input)
        return torch.sigmoid(mod_raw + self.mod_bias)


class TrigramSpace(nn.Module):
    def __init__(self, text_dim=128, state_dim=176, yin_dim=64, yang_dim=64,
                 sphere_dim=3, n_gua=64, n_domains=8, input_dim=0,
                 gate_type="constrained", struct_dim=64,
                 coherence_mode="wave", enable_curiosity=False,
                 max_recursion_depth=5):
        super().__init__()
        self.text_dim = text_dim
        self.state_dim = state_dim
        self.n_domains = n_domains
        self.gate_type = gate_type
        self.struct_dim = struct_dim
        self.coherence_mode = coherence_mode
        self.enable_curiosity = enable_curiosity
        self.max_recursion_depth = max_recursion_depth
        self.input_dim = input_dim if input_dim > 0 else text_dim
        self.mirror_mode = (self.input_dim == state_dim)
        if self.mirror_mode:
            half = state_dim // 2
            self.bifurcator = YinYangBifurcator(state_dim, half, half)
            self.to_state = None
            if gate_type == "resonance_v2":
                self.penetration = ConstrainedPenetration(state_dim, dropout=0.3, noise_std=0.05)
                self.resonance_cavity = ResonanceCavity(
                    state_dim, n_domains, coherence_mode=coherence_mode)
            else:
                self.penetration = ConstrainedPenetration(state_dim, dropout=0.3, noise_std=0.05)
        else:
            self.bifurcator = YinYangBifurcator(text_dim, yin_dim, yang_dim)
            combined_dim = yin_dim + yang_dim
            self.to_state = nn.Linear(combined_dim, state_dim)
            self.penetration = None
        self.curvature = WuxingCurvatureGenerator(state_dim)
        self.sphere = BaguaSphereMapper(state_dim, sphere_dim)
        self.folder = HeluoInteractionFolder(state_dim, n_gua)
        self.gua_prototype = nn.Embedding(n_gua, state_dim)
        nn.init.normal_(self.gua_prototype.weight, std=0.02)
        self.domain_classifier = DomainClassifier(state_dim, n_domains)
        self._initialized_from_v53 = False

    def initialize_from_v53(self, base_tp_weight, base_tp_bias, v53_gua_proto):
        with torch.no_grad():
            if v53_gua_proto.shape == (64, self.state_dim):
                self.gua_prototype.weight.copy_(v53_gua_proto)
        self._initialized_from_v53 = True

    def forward(self, pooled):
        yang, yin, alpha = self.bifurcator(pooled)
        combined = torch.cat([yang, yin], dim=-1)
        if self.mirror_mode:
            state = self.penetration(combined)
        else:
            state = self.to_state(combined)
        curved = self.curvature(state)
        sphere_result = self.sphere(curved)
        proto = self.gua_prototype.weight
        if self.gate_type == "resonance_v2" and hasattr(self, 'resonance_cavity'):
            coherence = self.resonance_cavity.compute_coherence(
                curved, self.domain_classifier(curved))
            wave_feedback = self.resonance_cavity.compute_wave_feedback(
                curved, self.domain_classifier(curved))
            folder_result = self.folder(
                curved, proto, coherence=coherence, wave_feedback=wave_feedback)
            folded = folder_result["folded"]
        else:
            folder_result = self.folder(curved, proto)
            folded = folder_result["folded"] if isinstance(folder_result, dict) else folder_result
        proto_norm = F.normalize(proto, dim=-1)
        sim = torch.matmul(F.normalize(folded, dim=-1), proto_norm.T)
        domain_logits = self.domain_classifier(folded)
        result = {
            "yang": yang, "yin": yin, "bifurcation_alpha": alpha,
            "state": state, "curved_state": curved,
            "sphere_coord": sphere_result["sphere_coord"],
            "xiantian_sim": sphere_result["xiantian_sim"],
            "houtian_sim": sphere_result["houtian_sim"],
            "combined_sim": sphere_result["combined_sim"],
            "flow_weight": sphere_result["flow_weight"],
            "folded": folded,
            "gua_similarity": sim,
            "gua_top1_idx": sim.argmax(dim=-1),
            "gua_top1_score": sim.max(dim=-1).values,
            "domain_logits": domain_logits,
            "domain_probs": F.log_softmax(domain_logits, dim=-1),
        }
        if self.gate_type == "resonance_v2" and hasattr(self, 'resonance_cavity'):
            coherence_final = self.resonance_cavity.compute_coherence(folded, domain_logits)
            modulation = self.resonance_cavity.compute_modulation(coherence_final, domain_logits)
            result["cavity_coherence"] = coherence_final
            result["cavity_modulation"] = modulation.squeeze(-1)
            result["wave_feedback"] = self.resonance_cavity.compute_wave_feedback(
                folded, domain_logits)
        return result

    def get_bagua_affinity(self, pooled):
        result = self.forward(pooled)
        combined_sim = result["combined_sim"]
        if combined_sim.dim() == 1:
            combined_sim = combined_sim.unsqueeze(0)
        bagua_scores = {}
        for i, name in enumerate(BAGUA_NAMES):
            bagua_scores[name] = combined_sim[0, i].item()
        best_gua = max(bagua_scores, key=bagua_scores.get)
        return {
            "best_gua": best_gua,
            "scores": bagua_scores,
            "wuxing": GUA_WUXING.get(best_gua, "未知"),
            "sphere_coord": result["sphere_coord"][0].detach(),
            "bifurcation_alpha": result["bifurcation_alpha"][0].item(),
        }


# ==============================================================================
# YiJingV53Foundation - Main Model
# ==============================================================================

class YiJingV53Foundation(nn.Module):
    """
    V53 Foundation Model for I Ching analysis.

    Dual-pathway architecture:
      - Text pathway: TextEncoder (128d) -> TextProj (176d)
      - Symbol pathway: SparseInput (176d)
      - Fusion: Gated combination -> HeLuoLadderNetwork -> Output Heads

    Three divination methods supported:
      - Traditional (周易): Palace-based analysis with Six Relatives
      - Plum Blossom (梅花): Image-number based analysis
      - Six Lines (六爻): Najia-based analysis with Six Spirits
    """

    def __init__(self, vocab_size=3000, text_dim=128, state_dim=176, hidden_dim=320,
                 ladder_layers=6, T=7, num_heads=8, dropout=0.2,
                 lora_rank=8, lora_alpha=0.1, enable_lora=True, gua_embed_dim=32,
                 moco_queue_size=4096, moco_momentum=0.999):
        super().__init__()
        self.state_dim = state_dim
        self.moco_queue_size = moco_queue_size
        self.moco_momentum = moco_momentum

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=text_dim,
            num_heads=4, num_layers=2, max_seq=MAX_SEQ, dropout=dropout)
        self.momentum_text_encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=text_dim,
            num_heads=4, num_layers=2, max_seq=MAX_SEQ, dropout=dropout)
        for param in self.momentum_text_encoder.parameters():
            param.requires_grad = False

        self.method_embed = nn.Embedding(3, 20)
        self.text_proj = nn.Linear(text_dim, state_dim)
        self.momentum_text_proj = nn.Linear(text_dim, state_dim)
        for param in self.momentum_text_proj.parameters():
            param.requires_grad = False

        self.gua_prototype = nn.Embedding(64, state_dim)
        nn.init.normal_(self.gua_prototype.weight, std=0.02)
        self.text_gate = nn.Sequential(nn.Linear(state_dim * 2, state_dim), nn.Sigmoid())

        self.heluo_ladder = HeLuoLadderNetwork(
            input_dim=state_dim + 20, state_dim=state_dim, hidden_dim=hidden_dim,
            num_layers=ladder_layers, T=T, dropout=dropout, gua_embed_dim=gua_embed_dim)

        self.head_traditional = OutputHeadV38(
            state_dim, dropout=dropout, lora_rank=lora_rank,
            lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.head_meihua = OutputHeadV38(
            state_dim, dropout=dropout, lora_rank=lora_rank,
            lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.head_liuyao = OutputHeadV38(
            state_dim, dropout=dropout, lora_rank=lora_rank,
            lora_alpha=lora_alpha, enable_lora=enable_lora)

        self.method_fusion = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlm_head = MLMHead(hidden_dim=64, vocab_size=vocab_size)

        self.register_buffer('moco_queue', torch.randn(moco_queue_size, state_dim))
        self.moco_queue = F.normalize(self.moco_queue, p=2, dim=-1)
        self.register_buffer('moco_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(
                self.text_encoder.parameters(),
                self.momentum_text_encoder.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (
                1.0 - self.moco_momentum)
        for param_q, param_k in zip(
                self.text_proj.parameters(),
                self.momentum_text_proj.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (
                1.0 - self.moco_momentum)

    def encode_text(self, text_ids):
        text_pooled, _ = self.text_encoder(text_ids)
        return self.text_proj(text_pooled)

    @torch.no_grad()
    def encode_text_momentum(self, text_ids):
        text_pooled, _ = self.momentum_text_encoder(text_ids)
        return self.momentum_text_proj(text_pooled)

    def forward(self, symbol_x, text_ids, method_idx, gua_idx,
                return_state=False, return_text_hidden=False):
        text_pooled, text_hidden = self.text_encoder(text_ids)
        text_feat = self.text_proj(text_pooled)
        mv = self.method_embed(method_idx)
        gate = self.text_gate(torch.cat([symbol_x, text_feat], dim=-1))
        fused_x = symbol_x + gate * text_feat
        c = torch.cat([fused_x, mv], dim=1)
        if return_state:
            features, state = self.heluo_ladder(c, gua_idx, return_state=True)
        else:
            features = self.heluo_ladder(c, gua_idx, return_state=False)
            state = None
        fused, _ = self.method_fusion(
            features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
        features = features + fused.squeeze(0)
        outputs = {
            'traditional': self.head_traditional(features, method_name='traditional'),
            'meihua': self.head_meihua(features, method_name='meihua'),
            'liuyao': self.head_liuyao(features, method_name='liuyao'),
            'text_feat': text_feat,
        }
        if return_text_hidden:
            outputs['mlm_logits'] = self.mlm_head(text_hidden)
        if return_state:
            return outputs, state
        return outputs

    def forward_pooled_text(self, symbol_x, text_feat, method_idx, gua_idx,
                            return_state=False):
        mv = self.method_embed(method_idx)
        gate = self.text_gate(torch.cat([symbol_x, text_feat], dim=-1))
        fused_x = symbol_x + gate * text_feat
        c = torch.cat([fused_x, mv], dim=1)
        if return_state:
            features, state = self.heluo_ladder(c, gua_idx, return_state=True)
        else:
            features = self.heluo_ladder(c, gua_idx, return_state=False)
            state = None
        fused, _ = self.method_fusion(
            features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
        features = features + fused.squeeze(0)
        outputs = {
            'traditional': self.head_traditional(features, method_name='traditional'),
            'meihua': self.head_meihua(features, method_name='meihua'),
            'liuyao': self.head_liuyao(features, method_name='liuyao'),
        }
        if return_state:
            return outputs, state
        return outputs

    def retrieval_logits(self, text_emb, temperature=0.1):
        proto = self.gua_prototype.weight
        proto_norm = F.normalize(proto, p=2, dim=-1)
        text_norm = F.normalize(text_emb, p=2, dim=-1)
        return torch.mm(text_norm, proto_norm.t()) / temperature

    def forward_mlm(self, input_ids, attention_mask, labels):
        _, hidden_states = self.text_encoder(input_ids, attention_mask)
        mlm_logits = self.mlm_head(hidden_states)
        active_mask = (labels != -100)
        if active_mask.sum() == 0:
            return torch.tensor(0.0, device=input_ids.device), 0.0
        active_logits = mlm_logits[active_mask]
        active_labels = labels[active_mask]
        loss = F.cross_entropy(active_logits, active_labels)
        with torch.no_grad():
            acc = (active_logits.argmax(dim=-1) == active_labels).float().mean().item()
        return loss, acc


# ==============================================================================
# Model Loading & Inference Utilities
# ==============================================================================

def load_v53_model(weights_path, config_path=None, device='cpu'):
    """
    Load a V53 foundation model from safetensors weights.
    
    NOTE: Model Weights are NOT included in this public repository.
    You must obtain weights through separate authorization from SmallLoong Research.
    See LICENSE Section 2.2 for access procedures.

    Uses safetensors format (no pickle, no code execution risk).
    SHA256 verification is recommended before loading.

    Args:
        weights_path: Path to .safetensors weights file
        config_path: Path to yijing_v53_config.json (auto-detected if None)
        device: Target device ('cpu', 'cuda', etc.)

    Returns:
        model: Loaded YiJingV53Foundation model

    Example:
        # After obtaining authorized weights:
        model = load_v53_model("yijing_v53_foundation.safetensors")
    """
    if config_path is None:
        import os
        dir_path = os.path.dirname(os.path.abspath(weights_path))
        candidate = os.path.join(dir_path, "yijing_v53_config.json")
        if os.path.exists(candidate):
            config_path = candidate

    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        vocab_size = config.get("vocab_size", 8145)
    else:
        vocab_size = 8145

    from safetensors.torch import load_file as safetensors_load
    model_state = safetensors_load(weights_path)

    model = YiJingV53Foundation(vocab_size=vocab_size)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def verify_sha256(weights_path, hash_path=None):
    """
    Verify the SHA256 integrity of a safetensors file.

    Args:
        weights_path: Path to .safetensors file
        hash_path: Path to .sha256 file (auto-detected if None)

    Returns:
        bool: True if hash matches, False otherwise
    """
    import hashlib
    import os

    if hash_path is None:
        hash_path = weights_path.replace(".safetensors", ".sha256")

    if not os.path.exists(hash_path):
        print(f"[WARN] Hash file not found: {hash_path}")
        return False

    with open(hash_path, "r") as f:
        expected_hash = f.read().strip().split()[0]

    sha = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual_hash = sha.hexdigest()

    if actual_hash == expected_hash:
        print(f"[OK] SHA256 verified: {actual_hash[:16]}...")
        return True
    else:
        print(f"[FAIL] SHA256 mismatch!")
        print(f"  Expected: {expected_hash[:16]}...")
        print(f"  Actual:   {actual_hash[:16]}...")
        return False


def predict_gua(model, text_ids, gua_idx, method='traditional', device='cpu'):
    """
    Run a prediction for a given hexagram.

    Args:
        model: YiJingV53Foundation model instance
        text_ids: Tokenized text input tensor (1, seq_len)
        gua_idx: Hexagram index (0-63)
        method: Divination method ('traditional', 'meihua', 'liuyao')
        device: Device to run on

    Returns:
        dict with prediction outputs for the specified method
    """
    method_idx = METHOD_MAP.get(method, 0)
    method_tensor = torch.tensor([method_idx], dtype=torch.long, device=device)
    gua_tensor = torch.tensor([gua_idx], dtype=torch.long, device=device)
    symbol_x = torch.tensor(
        [sparse_expand_input(gua_idx)], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(symbol_x, text_ids, method_tensor, gua_tensor)

    return outputs[method]


def predict_with_trigram_space(model, text_ids, device='cpu'):
    """
    Run text through the TrigramSpace for DaoTi analysis.

    Args:
        model: YiJingV53Foundation model instance
        text_ids: Tokenized text input tensor (1, seq_len)
        device: Device to run on

    Returns:
        TrigramSpace result dict with gua_similarity, domain_logits, etc.
    """
    trigram = TrigramSpace(text_dim=TEXT_DIM, state_dim=STATE_DIM,
                           input_dim=STATE_DIM,
                           gate_type="resonance_v2", coherence_mode="separation")
    trigram.to(device)
    trigram.eval()

    with torch.no_grad():
        text_pooled, _ = model.text_encoder(text_ids)
        text_feat = model.text_proj(text_pooled)
        result = trigram(text_feat)

    return result


# ==============================================================================
# Architecture Verification (Dry-Run, No Weights Required)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("YiJing V53 Foundation Model — Architecture Reference")
    print("=" * 70)
    print()
    print("NOTE: This is a REFERENCE IMPLEMENTATION only.")
    print("Model weights are NOT included in this public repository.")
    print("See LICENSE Section 2.2 for weight access procedures.")
    print()

    model = YiJingV53Foundation(vocab_size=3000)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  TextEncoder:         {sum(p.numel() for p in model.text_encoder.parameters()):,}")
    print(f"  HeLuoLadder:         {sum(p.numel() for p in model.heluo_ladder.parameters()):,}")
    print(f"  OutputHeads (x3):    {sum(p.numel() for p in model.head_traditional.parameters()):,} each")

    B, L = 2, 32
    symbol_x = torch.tensor(
        [sparse_expand_input(0), sparse_expand_input(1)], dtype=torch.float32)
    text_ids = torch.randint(1, 3000, (B, L))
    method_idx = torch.tensor([0, 1], dtype=torch.long)
    gua_idx = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(symbol_x, text_ids, method_idx, gua_idx)

    print(f"\nForward pass (random init): OK")
    print(f"  Traditional output keys: {list(outputs['traditional'].keys())}")
    print(f"  Palace logits shape: {outputs['traditional']['palace'].shape}")
    print(f"  Liuqin logits shape: {outputs['traditional']['liuqin'].shape}")
    print(f"  Text feature shape: {outputs['text_feat'].shape}")

    trigram = TrigramSpace(text_dim=TEXT_DIM, state_dim=STATE_DIM,
                           input_dim=STATE_DIM,
                           gate_type="resonance_v2", coherence_mode="separation")
    with torch.no_grad():
        text_pooled, _ = model.text_encoder(text_ids)
        text_feat = model.text_proj(text_pooled)
        ts_result = trigram(text_feat)
    print(f"\nTrigramSpace forward pass (random init): OK")
    print(f"  Gua similarity shape: {ts_result['gua_similarity'].shape}")
    print(f"  Domain logits shape: {ts_result['domain_logits'].shape}")

    print("\n" + "=" * 70)
    print("Architecture integrity verified.")
    print("To run inference: obtain authorized weights, then use load_v53_model().")
    print("=" * 70)
