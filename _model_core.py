"""
DaoTi V53 — Model Architecture (Inference Build)
==================================================
Copyright (c) 2025 DaoTi Research. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

--- Additional Terms (DaoTi Research License v1.0) ---
The model weights and architecture code provided herein are intended for
research and non-commercial use only. Commercial use, redistribution of
model weights, and reverse engineering of the training methodology are
strictly prohibited without explicit written permission.

NOTE: This file defines the model structure for inference only. All
hyperparameters are loaded from the saved model state_dict at runtime.
The default values below are placeholders and do not reflect the actual
training configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from _constants import (
    MAX_SEQ, STATE_DIM,
    XIAN_TIAN_MAP, HOU_TIAN_MAP, GUA_TRIGRAM,
    WUXING_SHENGKE_MATRIX, LIUQIN_MAP,
)

_DROPOUT_TEXT = 0.1
_DROPOUT_CORE = 0.2
_DROPOUT_PHYSICS = 0.1
_INTER_STEP_DROPOUT = 0.1
_LAYER_DROPOUT_SCALE = 0.5
_ENCODER_DROPOUT_SCALE = 0.8
_LORA_RANK = 8
_LORA_ALPHA = 0.1
_INIT_STD = 0.02


class _SeqEncoder(nn.Module):
    def __init__(self, vocab_size=3000, embed_dim=64, hidden_dim=128, num_heads=4, num_layers=2, max_seq=256, dropout=_DROPOUT_TEXT):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, text_ids, mask=None):
        B, L = text_ids.shape
        positions = torch.arange(L, device=text_ids.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(text_ids) + self.pos_embed(positions)
        h = self.layer_norm(h)
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        h = self.transformer(h, src_key_padding_mask=(text_ids == 0))
        non_pad = (text_ids != 0).unsqueeze(-1).float()
        h_pooled = (h * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)
        return self.pool_proj(h_pooled), h


class _MaskedHead(nn.Module):
    def __init__(self, hidden_dim=64, vocab_size=3000):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, hidden_states):
        return self.fc(hidden_states)


class CellBlock(nn.Module):
    def __init__(self, state_dim, hidden_dim, gua_embed_dim=32, dropout=_DROPOUT_CORE):
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
            nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.backward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.forward_gate = nn.Linear(state_dim * 2 + combined_dim, state_dim)
        self.backward_gate = nn.Linear(state_dim * 2 + combined_dim, state_dim)
        self.layer_dropout = nn.Dropout(dropout * _LAYER_DROPOUT_SCALE)

    def forward(self, state, gua_idx):
        gua_vec = self.gua_position_embed(gua_idx)
        xt_vec = self.xiantian_embed(self.xt_lut[gua_idx])
        ht_vec = self.houtian_embed(self.ht_lut[gua_idx])
        direction_vec = torch.cat([gua_vec, xt_vec, ht_vec], dim=-1)
        forward_pred = self.forward_net(state)
        forward_gate_val = torch.sigmoid(self.forward_gate(torch.cat([state, forward_pred, direction_vec], dim=-1)))
        forward_state = forward_gate_val * forward_pred + (1 - forward_gate_val) * state
        forward_state = self.layer_dropout(forward_state)
        backward_pred = self.backward_net(state)
        backward_gate_val = torch.sigmoid(self.backward_gate(torch.cat([state, backward_pred, direction_vec], dim=-1)))
        backward_state = backward_gate_val * backward_pred + (1 - backward_gate_val) * state
        backward_state = self.layer_dropout(backward_state)
        return forward_state, backward_state


class CoreEngine(nn.Module):
    def __init__(self, input_dim=196, state_dim=176, hidden_dim=320, num_layers=6, T=7, dropout=_DROPOUT_CORE, gua_embed_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T = T
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim), nn.Dropout(dropout * _ENCODER_DROPOUT_SCALE),
        ])
        self.ladder_layers = nn.ModuleList([
            CellBlock(state_dim, hidden_dim, gua_embed_dim=gua_embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim), nn.Dropout(dropout * _ENCODER_DROPOUT_SCALE),
        )
        self.multihead_attn = nn.MultiheadAttention(embed_dim=state_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x, gua_idx, return_state=False):
        state = self.input_encoder(x)
        for t in range(self.T):
            new_state = torch.zeros_like(state)
            for layer in self.ladder_layers:
                forward_state, backward_state = layer(state, gua_idx)
                new_state = new_state + (forward_state + backward_state) / 2.0
            if t < self.T - 1:
                new_state = F.dropout(new_state, p=_INTER_STEP_DROPOUT, training=self.training)
            state = new_state / self.num_layers
        attn_out, _ = self.multihead_attn(state.unsqueeze(1), state.unsqueeze(1), state.unsqueeze(1))
        output = self.output_decoder(attn_out.squeeze(1))
        output = self.norm(output)
        if return_state:
            return output, state
        return output


class RuleModule(nn.Module):
    def __init__(self, h=176, dropout=_DROPOUT_CORE):
        super().__init__()
        half_h = h // 2
        self.palace_wuxing_fc = nn.Linear(h, 5)
        self.dizhi_wuxing_fc = nn.Linear(h, 5)
        self.register_buffer('shengke_matrix', torch.from_numpy(WUXING_SHENGKE_MATRIX))
        self.liuqin_residual = nn.Sequential(
            nn.Linear(h + 6, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6),
        )
        self.liushen_rizhi_fc = nn.Linear(h, 10)
        self.liushen_residual = nn.Sequential(
            nn.Linear(h + 6 + 10, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6),
        )

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


class TaskHead(nn.Module):
    def __init__(self, h=176, dropout=_DROPOUT_CORE, lora_rank=_LORA_RANK, lora_alpha=_LORA_ALPHA, enable_lora=True):
        super().__init__()
        self.h = h
        self.enable_lora = enable_lora
        self.lora_alpha = lora_alpha
        half_h = h // 2
        self.shared_fc = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h),
        )
        self.semantic_prototypes = nn.Parameter(torch.randn(8, half_h) * _INIT_STD)
        self.classify_q = nn.Linear(h, half_h)
        self.classify_attn = nn.MultiheadAttention(embed_dim=half_h, num_heads=4, dropout=dropout, batch_first=True)
        self.wuxing_shengke = RuleModule(h, dropout)
        self.palace = nn.Linear(h, 8)
        self.tiangan = nn.Linear(h, 10)
        self.dizhi = nn.Linear(h, 12)
        self.wangxiang = nn.Linear(h + half_h, 5)
        self.biangua_yao = nn.Sequential(
            nn.Linear(h, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6),
        )
        if enable_lora:
            self.lora_A = nn.ModuleDict({
                name: nn.Linear(h, lora_rank, bias=False)
                for name in ['traditional', 'meihua', 'liuyao']
            })
            self.lora_B = nn.ModuleDict({
                name: nn.Linear(lora_rank, h, bias=False)
                for name in ['traditional', 'meihua', 'liuyao']
            })
            for m in ['traditional', 'meihua', 'liuyao']:
                nn.init.normal_(self.lora_A[m].weight, std=_INIT_STD)
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


class YiJingV53Foundation(nn.Module):
    def __init__(self, vocab_size=3000, text_dim=128, state_dim=176, hidden_dim=320,
                 ladder_layers=6, T=7, num_heads=8, dropout=_DROPOUT_CORE,
                 lora_rank=_LORA_RANK, lora_alpha=_LORA_ALPHA, enable_lora=True, gua_embed_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.text_encoder = _SeqEncoder(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=text_dim,
            num_heads=4, num_layers=2, max_seq=MAX_SEQ, dropout=dropout,
        )
        self.method_embed = nn.Embedding(3, 20)
        self.text_proj = nn.Linear(text_dim, state_dim)
        self.gua_prototype = nn.Embedding(64, state_dim)
        nn.init.normal_(self.gua_prototype.weight, std=_INIT_STD)
        self.text_gate = nn.Sequential(nn.Linear(state_dim * 2, state_dim), nn.Sigmoid())
        self.heluo_ladder = CoreEngine(
            input_dim=state_dim + 20, state_dim=state_dim, hidden_dim=hidden_dim,
            num_layers=ladder_layers, T=T, dropout=dropout, gua_embed_dim=gua_embed_dim,
        )
        self.head_traditional = TaskHead(state_dim, dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.head_meihua = TaskHead(state_dim, dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.head_liuyao = TaskHead(state_dim, dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.method_fusion = nn.MultiheadAttention(embed_dim=state_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlm_head = _MaskedHead(hidden_dim=64, vocab_size=vocab_size)

    def encode_text(self, text_ids):
        text_pooled, _ = self.text_encoder(text_ids)
        return self.text_proj(text_pooled)

    def forward(self, symbol_x, text_ids, method_idx, gua_idx, return_state=False, return_text_hidden=False):
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
        fused, _ = self.method_fusion(features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
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


class PhysicsInputEncoder(nn.Module):
    def __init__(self, input_dim, state_dim=176, hidden_dim=128, dropout=_DROPOUT_PHYSICS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class SpectrumRegressionHead(nn.Module):
    def __init__(self, state_dim=176, output_dim=100, hidden_dim=256, dropout=_DROPOUT_PHYSICS):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.regressor(x)


class PhysicsAdapterModel(nn.Module):
    def __init__(self, daoti_model, input_dim, output_dim, state_dim=176, freeze_daoti=True):
        super().__init__()
        self.daoti = daoti_model
        self.input_encoder = PhysicsInputEncoder(input_dim, state_dim)
        self.regression_head = SpectrumRegressionHead(state_dim, output_dim)
        if freeze_daoti:
            for param in self.daoti.parameters():
                param.requires_grad = False

    def forward(self, physics_params):
        state = self.input_encoder(physics_params)
        with torch.no_grad():
            gua_idx = torch.zeros(physics_params.size(0), dtype=torch.long, device=physics_params.device)
            method_idx = torch.zeros(physics_params.size(0), dtype=torch.long, device=physics_params.device)
            dummy_text = torch.ones(physics_params.size(0), MAX_SEQ, dtype=torch.long, device=physics_params.device)
            text_feat = self.daoti.encode_text(dummy_text)
            gate = self.daoti.text_gate(torch.cat([state, text_feat], dim=-1))
            fused_x = state + gate * text_feat
            method_vec = self.daoti.method_embed(method_idx)
            c = torch.cat([fused_x, method_vec], dim=1)
            features = self.daoti.heluo_ladder(c, gua_idx)
            fused, _ = self.daoti.method_fusion(
                features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0),
            )
            features = features + fused.squeeze(0)
        spectrum = self.regression_head(features)
        return spectrum
