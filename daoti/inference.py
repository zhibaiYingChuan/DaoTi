"""
DaoTi V53 Foundation Model — Inference Interface
==================================================
道体基座推理接口（公开发布版）。

公开接口:
  - load_daoti()         加载模型
  - predict()            执行推理
  - generate_response()  生成自然语言回答
  - compute_coherence()  计算相干性
  - verify_sha256()      校验权重完整性
  - load_adapter()       加载领域适配器
  - load_physics_adapter()  加载物理适配器
  - predict_physics()    物理参数预测

常量与数据:
  - GUA_64, BA_GONG, GUA_WUXING, GUA_TRIGRAM 等
  - sparse_expand_input(), find_palace() 等工具函数
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import os
import hashlib

from daoti._constants import *


def _get_model_class():
    from daoti._model_core import YiJingV53Foundation
    return YiJingV53Foundation


def verify_sha256(weights_path, hash_path=None):
    if hash_path is None: hash_path = weights_path+".sha256"
    if not os.path.exists(hash_path):
        print(f"[WARN] Hash file not found: {hash_path}")
        return False
    with open(hash_path, "r") as f: expected_hash = f.read().strip().split()[0]
    sha = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): sha.update(chunk)
    actual_hash = sha.hexdigest()
    if actual_hash == expected_hash:
        print(f"[OK] SHA256 verified: {actual_hash[:16]}...")
        return True
    print(f"[FAIL] SHA256 mismatch!\n  Expected: {expected_hash[:16]}...\n  Actual:   {actual_hash[:16]}...")
    return False

def load_daoti(weights_path, device='cpu'):
    """
    Load DaoTi V53 model from .pt weights file.

    Args:
        weights_path: Path to yijing_v53_daoti.pt
        device: 'cpu' or 'cuda'

    Returns:
        Model ready for inference
    """
    import json
    dir_path = os.path.dirname(os.path.abspath(weights_path))
    config_path = os.path.join(dir_path, "yijing_v53_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f: config = json.load(f)
        vocab_size = config.get("vocab_size", 8145)
    else:
        vocab_size = 8145
    YiJingV53Foundation = _get_model_class()
    model = YiJingV53Foundation(vocab_size=vocab_size)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def load_adapter(model, adapter_path, device='cpu'):
    """
    Load a trained domain adapter into the model.

    Args:
        model: Loaded model
        adapter_path: Path to adapter .pt file
        device: 'cpu' or 'cuda'

    Returns:
        model with adapter weights applied
    """
    adapter_data = torch.load(adapter_path, map_location=device, weights_only=False)
    weights = adapter_data.get('weights', {})
    domain = adapter_data.get('domain', 'unknown')
    method = adapter_data.get('method', 'traditional')

    model_state = model.state_dict()
    loaded = 0
    for name, tensor in weights.items():
        if name in model_state:
            model_state[name] = tensor.to(device)
            loaded += 1

    model.load_state_dict(model_state, strict=False)
    print(f"[OK] Adapter loaded: domain='{domain}', method='{method}', {loaded} weight tensors")
    return model

def load_physics_adapter(adapter_path, device='cpu'):
    """
    Load a trained physics adapter for spectrum prediction.

    Args:
        adapter_path: Path to physics adapter .pt file
        device: 'cpu' or 'cuda'

    Returns:
        dict with adapter data including weights, norm_stats, input_dim, output_dim
    """
    adapter_data = torch.load(adapter_path, map_location=device, weights_only=False)
    if adapter_data.get('type') != 'physics_adapter':
        print(f"[WARN] Not a physics adapter file: {adapter_path}")
        return None
    input_dim = adapter_data.get('input_dim', 0)
    output_dim = adapter_data.get('output_dim', 0)
    norm_stats = adapter_data.get('norm_stats', None)
    if norm_stats:
        for k, v in norm_stats.items():
            if isinstance(v, list):
                norm_stats[k] = torch.tensor(v, dtype=torch.float32, device=device)
    print(f"[OK] Physics adapter loaded: input_dim={input_dim}, output_dim={output_dim}")
    return adapter_data

def predict_physics(daoti_model, adapter_data, physics_params, device='cpu'):
    """
    Predict spectrum from physics parameters using a trained physics adapter.

    Args:
        daoti_model: Loaded model
        adapter_data: Adapter data dict from load_physics_adapter()
        physics_params: Tensor of shape (batch, input_dim) or (input_dim,)
        device: 'cpu' or 'cuda'

    Returns:
        dict with:
            - 'spectrum': predicted spectrum values (denormalized)
            - 'spectrum_norm': normalized spectrum values
    """
    from _model_core import PhysicsAdapterModel

    if physics_params.dim() == 1:
        physics_params = physics_params.unsqueeze(0)

    input_dim = adapter_data['input_dim']
    output_dim = adapter_data['output_dim']
    norm_stats = adapter_data.get('norm_stats', {})

    adapter_model = PhysicsAdapterModel(
        daoti_model, input_dim, output_dim,
        state_dim=STATE_DIM, freeze_daoti=True
    ).to(device)

    weights = adapter_data.get('weights', {})
    adapter_state = adapter_model.state_dict()
    for name, tensor in weights.items():
        if name in adapter_state:
            adapter_state[name] = tensor.to(device)
    adapter_model.load_state_dict(adapter_state, strict=False)
    adapter_model.eval()

    params_norm = physics_params.to(device)
    if norm_stats and 'params_mean' in norm_stats:
        params_mean = norm_stats['params_mean'].to(device)
        params_std = norm_stats['params_std'].to(device)
        params_norm = (physics_params.to(device) - params_mean) / params_std

    with torch.no_grad():
        spectrum_norm = adapter_model(params_norm)

    spectrum = spectrum_norm
    if norm_stats and 'spectrum_mean' in norm_stats:
        spec_mean = norm_stats['spectrum_mean'].to(device)
        spec_std = norm_stats['spectrum_std'].to(device)
        spectrum = spectrum_norm * spec_std + spec_mean

    return {
        'spectrum': spectrum.cpu(),
        'spectrum_norm': spectrum_norm.cpu(),
    }

def predict(model, text_ids, gua_idx, method='traditional', device='cpu'):
    """
    Run divination prediction.

    Args:
        model: Loaded model
        text_ids: Tokenized text input, shape (1, seq_len), int tensor
        gua_idx: Hexagram index 0-63
        method: 'traditional' (周易), 'meihua' (梅花), or 'liuyao' (六爻)
        device: 'cpu' or 'cuda'

    Returns:
        dict with prediction outputs and coherence score
    """
    method_idx = METHOD_MAP.get(method, 0)
    method_tensor = torch.tensor([method_idx], dtype=torch.long, device=device)
    gua_tensor = torch.tensor([gua_idx], dtype=torch.long, device=device)
    symbol_x = torch.tensor([sparse_expand_input(gua_idx)], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(symbol_x, text_ids.to(device), method_tensor, gua_tensor)
    result = {k: v.cpu() for k, v in outputs[method].items()}
    result['coherence'] = compute_coherence(model, text_ids, gua_idx, device)
    return result

def compute_coherence(model, text_ids, gua_idx, device='cpu'):
    """
    Compute self-calibrating quality signal based on resonance cavity coherence.
    Higher coherence = model is more confident in its prediction.

    Returns:
        float: coherence score in [0, 1]
    """
    with torch.no_grad():
        text_feat = model.encode_text(text_ids.to(device))
        proto = model.gua_prototype.weight
        proto_n = F.normalize(proto, p=2, dim=-1)
        feat_n = F.normalize(text_feat, p=2, dim=-1)
        similarity = torch.mm(feat_n, proto_n.t()).squeeze()
        top_sim = similarity.max().item()
        coherence = max(0.0, min(1.0, top_sim))
    return coherence

def generate_response(model, text_ids, gua_idx, method='traditional', device='cpu', coherence_threshold=0.3):
    """
    Generate a structured natural language response via retrieval-augmented generation.

    Args:
        model: Loaded model
        text_ids: Tokenized text input, shape (1, seq_len), int tensor
        gua_idx: Hexagram index 0-63
        method: 'traditional', 'meihua', or 'liuyao'
        device: 'cpu' or 'cuda'
        coherence_threshold: If coherence < threshold, append low-confidence warning

    Returns:
        dict with:
            - 'response': str, composed natural language response
            - 'coherence': float, self-calibrating quality signal
            - 'low_confidence': bool, whether coherence is below threshold
            - 'details': dict, structured breakdown of the reasoning chain
    """
    pred = predict(model, text_ids, gua_idx, method=method, device=device)
    coherence = pred['coherence']

    PALACE_NAMES  = ["乾宫","坤宫","震宫","巽宫","坎宫","离宫","艮宫","兑宫"]
    TIANGAN_NAMES = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
    DIZHI_NAMES   = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]
    LIUQIN_NAMES  = ["父母","兄弟","子孙","妻财","官鬼","空亡"]
    LIUSHEN_NAMES = ["青龙","朱雀","勾陈","螣蛇","白虎","玄武"]
    WANGXIANG_N   = ["旺","相","休","囚","死"]

    gua_name = GUA_64[gua_idx]
    palace_name = PALACE_NAMES[pred['palace'].argmax().item()]
    palace_conf = F.softmax(pred['palace'], dim=-1).max().item()
    tiangan_name = TIANGAN_NAMES[pred['tiangan'].argmax().item()]
    dizhi_name = DIZHI_NAMES[pred['dizhi'].argmax().item()]
    liuqin_name = LIUQIN_NAMES[pred['liuqin'].argmax().item()]
    liushen_name = LIUSHEN_NAMES[pred['liushen'].argmax().item()]
    wangxiang_name = WANGXIANG_N[pred['wangxiang'].argmax().item()]
    yao_raw = pred['biangua_yao'].squeeze()
    yao_p = torch.sigmoid(yao_raw).tolist()
    yao_v = [1 if p > 0.5 else 0 for p in yao_p]
    moving_yao = [i+1 for i, v in enumerate(yao_v) if v]

    gua_info = GUA_64_DETAIL.get(gua_name, {})
    palace_info = KNOWLEDGE_BASE.get(palace_name, {})
    liuqin_info = LIUQIN_KNOWLEDGE.get(liuqin_name, {})
    liushen_info = LIUSHEN_KNOWLEDGE.get(liushen_name, {})
    wangxiang_desc = WANGXIANG_KNOWLEDGE.get(wangxiang_name, "")

    lines = []
    lines.append(f"【{gua_name}】{gua_info.get('上下卦', '')}，属{palace_name}（{palace_info.get('五行','')}行）")
    lines.append("")
    lines.append(f"卦象要义：{palace_info.get('象义', '')}")
    lines.append(f"经典原文：{palace_info.get('经典', '')}")
    lines.append("")
    lines.append(f"推理结果：")
    lines.append(f"  天干地支：{tiangan_name}{dizhi_name}")
    lines.append(f"  六亲持世：{liuqin_name}（{liuqin_info.get('象义','')}）")
    lines.append(f"  六神临爻：{liushen_name}（{liushen_info.get('象义','')}）")
    lines.append(f"  旺相休囚：{wangxiang_name}（{wangxiang_desc}）")
    if moving_yao:
        lines.append(f"  动爻：第{'、'.join(str(y) for y in moving_yao)}爻")
    else:
        lines.append(f"  动爻：无（静卦）")
    lines.append("")
    lines.append(f"综合解读：")
    lines.append(f"  {palace_info.get('解读', '')}")
    lines.append(f"  {liuqin_info.get('解读', '')}")
    lines.append(f"  {liushen_info.get('解读', '')}")

    low_conf = coherence < coherence_threshold
    if low_conf:
        lines.append("")
        lines.append(f"⚠️ 置信度较低（{coherence:.2f}），建议结合实际情况综合判断。")

    details = {
        'gua_name': gua_name,
        'palace': palace_name,
        'palace_confidence': round(palace_conf, 4),
        'tiangan': tiangan_name,
        'dizhi': dizhi_name,
        'liuqin': liuqin_name,
        'liushen': liushen_name,
        'wangxiang': wangxiang_name,
        'moving_yao': moving_yao,
        'coherence': round(coherence, 4),
    }

    return {
        'response': '\n'.join(lines),
        'coherence': coherence,
        'low_confidence': low_conf,
        'details': details,
    }
