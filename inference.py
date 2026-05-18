import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import os
import hashlib

MAX_SEQ = 256
STATE_DIM = 176
TEXT_DIM = 128

GUA_64 = [
    "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
    "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离","咸","恒","遁","大壮","晋","明夷",
    "家人","睽","蹇","解","损","益","夬","姤","萃","升","困","井",
    "革","鼎","震","艮","渐","归妹","丰","旅","巽","兑","涣","节",
    "中孚","小过","既济","未济",
]

BA_GONG = {
    "乾宫":["乾","姤","遁","否","观","剥","晋","大有"],
    "坤宫":["坤","复","临","泰","大壮","夬","需","比"],
    "震宫":["震","豫","解","恒","升","井","大过","随"],
    "巽宫":["巽","小畜","家人","益","无妄","噬嗑","颐","蛊"],
    "坎宫":["坎","节","屯","既济","革","丰","明夷","师"],
    "离宫":["离","旅","鼎","未济","蒙","涣","讼","同人"],
    "艮宫":["艮","贲","大畜","损","睽","履","中孚","渐"],
    "兑宫":["兑","困","萃","咸","蹇","谦","小过","归妹"],
}

GUA_WUXING = {"乾":"金","兑":"金","坤":"土","艮":"土","震":"木","巽":"木","坎":"水","离":"火"}

XIAN_TIAN_MAP = {
    0:0,1:7,2:4,3:6,4:5,5:0,6:5,7:7,8:1,9:1,10:7,11:0,12:2,13:0,14:6,15:4,
    16:3,17:5,18:7,19:5,20:2,21:2,22:6,23:4,24:0,25:6,26:3,27:1,28:5,29:2,30:6,31:3,
    32:5,33:0,34:2,35:2,36:5,37:1,38:8,39:5,40:6,41:4,42:0,43:5,44:6,45:4,46:6,47:5,
    48:2,49:5,50:3,51:6,52:5,53:1,54:2,55:8,56:5,57:1,58:3,59:0,60:6,61:8,62:0,63:9,
}

HOU_TIAN_MAP = {
    0:5,1:1,2:2,3:8,4:0,5:5,6:0,7:1,8:3,9:6,10:1,11:5,12:9,13:5,14:8,15:2,
    16:2,17:3,18:1,19:3,20:9,21:9,22:8,23:2,24:5,25:8,26:2,27:6,28:0,29:9,30:6,31:8,
    32:3,33:5,34:9,35:9,36:3,37:6,38:8,39:0,40:8,41:3,42:5,43:3,44:6,45:3,46:6,47:3,
    48:9,49:3,50:2,51:8,52:3,53:6,54:9,55:8,56:3,57:6,58:3,59:0,60:6,61:8,62:0,63:9,
}

GUA_TRIGRAM = {
    0:("乾","乾"),1:("坤","坤"),2:("坎","震"),3:("艮","坎"),4:("坎","乾"),5:("乾","坎"),
    6:("坤","坎"),7:("坤","坎"),8:("巽","乾"),9:("乾","兑"),10:("坤","乾"),11:("乾","坤"),
    12:("乾","离"),13:("离","乾"),14:("艮","坤"),15:("坤","震"),16:("兑","震"),17:("艮","巽"),
    18:("坤","兑"),19:("坤","巽"),20:("离","震"),21:("艮","离"),22:("艮","坤"),23:("坤","震"),
    24:("乾","震"),25:("艮","乾"),26:("艮","震"),27:("兑","巽"),28:("坎","坎"),29:("离","离"),
    30:("兑","艮"),31:("震","巽"),32:("巽","乾"),33:("震","乾"),34:("离","坤"),35:("坤","离"),
    36:("巽","离"),37:("离","兑"),38:("坎","艮"),39:("震","坎"),40:("艮","兑"),41:("巽","震"),
    42:("乾","兑"),43:("乾","巽"),44:("兑","坤"),45:("坤","巽"),46:("兑","坎"),47:("巽","坎"),
    48:("兑","离"),49:("离","巽"),50:("震","震"),51:("艮","艮"),52:("艮","巽"),53:("震","兑"),
    54:("离","震"),55:("离","艮"),56:("巽","巽"),57:("兑","兑"),58:("巽","坎"),59:("坎","兑"),
    60:("巽","兑"),61:("艮","震"),62:("坎","离"),63:("离","坎"),
}

WUXING_IDX = {"金":0,"木":1,"水":2,"火":3,"土":4}
LIUQIN_MAP = {"父母":0,"兄弟":1,"子孙":2,"妻财":3,"官鬼":4,"空亡":5}
PALACE_MAP = {"乾宫":0,"坤宫":1,"震宫":2,"巽宫":3,"坎宫":4,"离宫":5,"艮宫":6,"兑宫":7}
METHOD_MAP = {"周易":0,"六爻":1,"梅花":2}
BAGUA_NAMES = ["乾","坤","震","巽","坎","离","艮","兑"]
WUXING_NAMES = ["木","火","土","金","水"]
WUXING_SHENG = {"木":"火","火":"土","土":"金","金":"水","水":"木"}
WUXING_KE = {"木":"土","火":"金","土":"水","金":"木","水":"火"}

def _build_wuxing_shengke_matrix():
    def _get_liuqin(p,n):
        if p==n: return "兄弟"
        if (p,n) in [("木","水"),("火","木"),("土","火"),("金","土"),("水","金")]: return "子孙"
        if (p,n) in [("木","金"),("火","木"),("土","水"),("金","火"),("水","土")]: return "官鬼"
        if (p,n) in [("木","土"),("火","金"),("土","木"),("金","水"),("水","火")]: return "妻财"
        return "父母"
    matrix = np.zeros((5,5,6), dtype=np.float32)
    for pi, pn in enumerate(["金","木","水","火","土"]):
        for ni, nn_wx in enumerate(["金","木","水","火","土"]):
            lq = _get_liuqin(pn, nn_wx)
            matrix[pi, ni, LIUQIN_MAP[lq]] = 1.0
    return matrix

WUXING_SHENGKE_MATRIX = _build_wuxing_shengke_matrix()

def gua_to_yao_bits(gi):
    return [float((gi>>j)&1) for j in range(5,-1,-1)]

def find_palace(g):
    for p, gs in BA_GONG.items():
        if g in gs: return p
    return "乾宫"

def sparse_expand_input(gi, yp=0, yi=0, ri=0, si=0, mi=0):
    x = np.zeros(STATE_DIM, dtype=np.float32)
    for j in range(6): x[j] = float(((gi>>(5-j))&1)*2-1)
    xt = XIAN_TIAN_MAP.get(gi,0)
    if xt<8: x[6+xt]=1.0
    ht = HOU_TIAN_MAP.get(gi,0)
    if ht<9: x[14+ht]=1.0
    upper_gua, lower_gua = GUA_TRIGRAM.get(gi,("乾","乾"))
    gua_wuxing = GUA_WUXING.get(upper_gua,"金")
    wx_idx = WUXING_IDX.get(gua_wuxing,0)
    x[23+wx_idx]=1.0
    palace = find_palace(GUA_64[gi])
    palace_idx = PALACE_MAP.get(palace,0)
    x[28+palace_idx]=1.0
    return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=3000, embed_dim=64, hidden_dim=128, num_heads=4, num_layers=2, max_seq=256, dropout=0.1):
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
        if mask is not None: h = h.masked_fill(mask.unsqueeze(-1)==0, 0.0)
        h = self.transformer(h, src_key_padding_mask=(text_ids==0))
        non_pad = (text_ids!=0).unsqueeze(-1).float()
        h_pooled = (h*non_pad).sum(dim=1)/non_pad.sum(dim=1).clamp(min=1)
        return self.pool_proj(h_pooled), h

class MLMHead(nn.Module):
    def __init__(self, hidden_dim=64, vocab_size=3000):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size))
    def forward(self, hidden_states): return self.fc(hidden_states)

class HeLuoLadderCell(nn.Module):
    def __init__(self, state_dim, hidden_dim, gua_embed_dim=32, dropout=0.2):
        super().__init__()
        self.gua_embed_dim = gua_embed_dim
        self.gua_position_embed = nn.Embedding(64, gua_embed_dim)
        self.xiantian_embed = nn.Embedding(8, gua_embed_dim//2)
        self.houtian_embed = nn.Embedding(9, gua_embed_dim//2)
        xt_lut = torch.zeros(64, dtype=torch.long); ht_lut = torch.zeros(64, dtype=torch.long)
        for i in range(64): xt_lut[i] = min(XIAN_TIAN_MAP.get(i,0), 7); ht_lut[i] = min(HOU_TIAN_MAP.get(i,0), 8)
        self.register_buffer('xt_lut', xt_lut); self.register_buffer('ht_lut', ht_lut)
        combined_dim = gua_embed_dim + gua_embed_dim//2 + gua_embed_dim//2
        self.forward_net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, state_dim))
        self.backward_net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, state_dim))
        self.forward_gate = nn.Linear(state_dim*2+combined_dim, state_dim)
        self.backward_gate = nn.Linear(state_dim*2+combined_dim, state_dim)
        self.layer_dropout = nn.Dropout(dropout*0.5)

    def forward(self, state, gua_idx):
        gua_vec = self.gua_position_embed(gua_idx)
        xt_vec = self.xiantian_embed(self.xt_lut[gua_idx])
        ht_vec = self.houtian_embed(self.ht_lut[gua_idx])
        direction_vec = torch.cat([gua_vec, xt_vec, ht_vec], dim=-1)
        forward_pred = self.forward_net(state)
        forward_gate_val = torch.sigmoid(self.forward_gate(torch.cat([state, forward_pred, direction_vec], dim=-1)))
        forward_state = forward_gate_val*forward_pred + (1-forward_gate_val)*state
        forward_state = self.layer_dropout(forward_state)
        backward_pred = self.backward_net(state)
        backward_gate_val = torch.sigmoid(self.backward_gate(torch.cat([state, backward_pred, direction_vec], dim=-1)))
        backward_state = backward_gate_val*backward_pred + (1-backward_gate_val)*state
        backward_state = self.layer_dropout(backward_state)
        return forward_state, backward_state

class HeLuoLadderNetwork(nn.Module):
    def __init__(self, input_dim=196, state_dim=176, hidden_dim=320, num_layers=6, T=7, dropout=0.2, gua_embed_dim=32):
        super().__init__()
        self.state_dim = state_dim; self.hidden_dim = hidden_dim; self.num_layers = num_layers; self.T = T
        self.input_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim), nn.Dropout(dropout*0.8))
        self.ladder_layers = nn.ModuleList([HeLuoLadderCell(state_dim, hidden_dim, gua_embed_dim=gua_embed_dim, dropout=dropout) for _ in range(num_layers)])
        self.output_decoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim), nn.Dropout(dropout*0.8))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=state_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x, gua_idx, return_state=False):
        state = self.input_encoder(x)
        for t in range(self.T):
            new_state = torch.zeros_like(state)
            for layer in self.ladder_layers:
                forward_state, backward_state = layer(state, gua_idx)
                new_state = new_state + (forward_state+backward_state)/2.0
            if t < self.T-1: new_state = F.dropout(new_state, p=0.1, training=self.training)
            state = new_state/self.num_layers
        attn_out, _ = self.multihead_attn(state.unsqueeze(1), state.unsqueeze(1), state.unsqueeze(1))
        output = self.output_decoder(attn_out.squeeze(1))
        output = self.norm(output)
        if return_state: return output, state
        return output

class WuxingShengkeModule(nn.Module):
    def __init__(self, h=176, dropout=0.2):
        super().__init__()
        half_h = h//2
        self.palace_wuxing_fc = nn.Linear(h, 5); self.dizhi_wuxing_fc = nn.Linear(h, 5)
        self.register_buffer('shengke_matrix', torch.from_numpy(WUXING_SHENGKE_MATRIX))
        self.liuqin_residual = nn.Sequential(nn.Linear(h+6, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6))
        self.liushen_rizhi_fc = nn.Linear(h, 10)
        self.liushen_residual = nn.Sequential(nn.Linear(h+6+10, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6))

    def forward(self, x):
        palace_wx_logits = self.palace_wuxing_fc(x); dizhi_wx_logits = self.dizhi_wuxing_fc(x)
        palace_wx = F.softmax(palace_wx_logits, dim=-1); dizhi_wx = F.softmax(dizhi_wx_logits, dim=-1)
        rule_liuqin = torch.einsum('bp,pnk,bn->bk', palace_wx, self.shengke_matrix, dizhi_wx)
        liuqin_res = self.liuqin_residual(torch.cat([x, rule_liuqin], dim=-1))
        liuqin_out = rule_liuqin + liuqin_res
        rizhi_logits = self.liushen_rizhi_fc(x)
        liushen_res = self.liushen_residual(torch.cat([x, rule_liuqin, rizhi_logits], dim=-1))
        return liuqin_out, liushen_res, palace_wx_logits, dizhi_wx_logits

class OutputHeadV38(nn.Module):
    def __init__(self, h=176, dropout=0.2, lora_rank=8, lora_alpha=0.1, enable_lora=True):
        super().__init__()
        self.h = h; self.enable_lora = enable_lora; self.lora_alpha = lora_alpha
        half_h = h//2
        self.shared_fc = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h))
        self.semantic_prototypes = nn.Parameter(torch.randn(8, half_h)*0.02)
        self.classify_q = nn.Linear(h, half_h)
        self.classify_attn = nn.MultiheadAttention(embed_dim=half_h, num_heads=4, dropout=dropout, batch_first=True)
        self.wuxing_shengke = WuxingShengkeModule(h, dropout)
        self.palace = nn.Linear(h, 8); self.tiangan = nn.Linear(h, 10); self.dizhi = nn.Linear(h, 12)
        self.wangxiang = nn.Linear(h+half_h, 5)
        self.biangua_yao = nn.Sequential(nn.Linear(h, half_h), nn.GELU(), nn.Dropout(dropout), nn.Linear(half_h, 6))
        if enable_lora:
            self.lora_A = nn.ModuleDict({name: nn.Linear(h, lora_rank, bias=False) for name in ['traditional','meihua','liuyao']})
            self.lora_B = nn.ModuleDict({name: nn.Linear(lora_rank, h, bias=False) for name in ['traditional','meihua','liuyao']})
            for m in ['traditional','meihua','liuyao']: nn.init.normal_(self.lora_A[m].weight, std=0.02); nn.init.zeros_(self.lora_B[m].weight)

    def forward(self, x, method_name='traditional'):
        half_h = self.h//2
        shared_features = self.shared_fc(x)
        if self.enable_lora and method_name in self.lora_A:
            lora_delta = self.lora_B[method_name](self.lora_A[method_name](shared_features))
            shared_features = shared_features + self.lora_alpha*lora_delta
        q = self.classify_q(shared_features).unsqueeze(1)
        prototypes = self.semantic_prototypes.unsqueeze(0).expand(q.size(0), -1, -1)
        classified, _ = self.classify_attn(q, prototypes, prototypes)
        classified = classified.squeeze(1)
        liuqin_out, liushen_out, palace_wx_logits, dizhi_wx_logits = self.wuxing_shengke(x)
        standard_out = torch.cat([shared_features, classified], dim=-1)
        return {'palace':self.palace(shared_features), 'tiangan':self.tiangan(shared_features), 'dizhi':self.dizhi(shared_features), 'liuqin':liuqin_out, 'liushen':liushen_out, 'wangxiang':self.wangxiang(standard_out), 'biangua_yao':self.biangua_yao(shared_features), 'palace_wuxing':palace_wx_logits, 'dizhi_wuxing':dizhi_wx_logits}

class YiJingV53Foundation(nn.Module):
    def __init__(self, vocab_size=3000, text_dim=128, state_dim=176, hidden_dim=320, ladder_layers=6, T=7, num_heads=8, dropout=0.2, lora_rank=8, lora_alpha=0.1, enable_lora=True, gua_embed_dim=32, moco_queue_size=4096, moco_momentum=0.999):
        super().__init__()
        self.state_dim = state_dim; self.moco_queue_size = moco_queue_size; self.moco_momentum = moco_momentum
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=64, hidden_dim=text_dim, num_heads=4, num_layers=2, max_seq=MAX_SEQ, dropout=dropout)
        self.momentum_text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=64, hidden_dim=text_dim, num_heads=4, num_layers=2, max_seq=MAX_SEQ, dropout=dropout)
        for param in self.momentum_text_encoder.parameters(): param.requires_grad = False
        self.method_embed = nn.Embedding(3, 20)
        self.text_proj = nn.Linear(text_dim, state_dim); self.momentum_text_proj = nn.Linear(text_dim, state_dim)
        for param in self.momentum_text_proj.parameters(): param.requires_grad = False
        self.gua_prototype = nn.Embedding(64, state_dim); nn.init.normal_(self.gua_prototype.weight, std=0.02)
        self.text_gate = nn.Sequential(nn.Linear(state_dim*2, state_dim), nn.Sigmoid())
        self.heluo_ladder = HeLuoLadderNetwork(input_dim=state_dim+20, state_dim=state_dim, hidden_dim=hidden_dim, num_layers=ladder_layers, T=T, dropout=dropout, gua_embed_dim=gua_embed_dim)
        self.head_traditional = OutputHeadV38(state_dim, dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.head_meihua = OutputHeadV38(state_dim, dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.head_liuyao = OutputHeadV38(state_dim, dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, enable_lora=enable_lora)
        self.method_fusion = nn.MultiheadAttention(embed_dim=state_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlm_head = MLMHead(hidden_dim=64, vocab_size=vocab_size)
        self.register_buffer('moco_queue', torch.randn(moco_queue_size, state_dim)); self.moco_queue = F.normalize(self.moco_queue, p=2, dim=-1)
        self.register_buffer('moco_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.text_encoder.parameters(), self.momentum_text_encoder.parameters()): param_k.data = param_k.data*self.moco_momentum + param_q.data*(1.0-self.moco_momentum)
        for param_q, param_k in zip(self.text_proj.parameters(), self.momentum_text_proj.parameters()): param_k.data = param_k.data*self.moco_momentum + param_q.data*(1.0-self.moco_momentum)

    def encode_text(self, text_ids):
        text_pooled, _ = self.text_encoder(text_ids)
        return self.text_proj(text_pooled)

    @torch.no_grad()
    def encode_text_momentum(self, text_ids):
        text_pooled, _ = self.momentum_text_encoder(text_ids)
        return self.momentum_text_proj(text_pooled)

    def forward(self, symbol_x, text_ids, method_idx, gua_idx, return_state=False, return_text_hidden=False):
        text_pooled, text_hidden = self.text_encoder(text_ids)
        text_feat = self.text_proj(text_pooled)
        mv = self.method_embed(method_idx)
        gate = self.text_gate(torch.cat([symbol_x, text_feat], dim=-1))
        fused_x = symbol_x + gate*text_feat
        c = torch.cat([fused_x, mv], dim=1)
        if return_state: features, state = self.heluo_ladder(c, gua_idx, return_state=True)
        else: features = self.heluo_ladder(c, gua_idx, return_state=False); state = None
        fused, _ = self.method_fusion(features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
        features = features + fused.squeeze(0)
        outputs = {'traditional':self.head_traditional(features, method_name='traditional'), 'meihua':self.head_meihua(features, method_name='meihua'), 'liuyao':self.head_liuyao(features, method_name='liuyao'), 'text_feat':text_feat}
        if return_text_hidden: outputs['mlm_logits'] = self.mlm_head(text_hidden)
        if return_state: return outputs, state
        return outputs

    def retrieval_logits(self, text_emb, temperature=0.1):
        proto = self.gua_prototype.weight
        proto_norm = F.normalize(proto, p=2, dim=-1)
        text_norm = F.normalize(text_emb, p=2, dim=-1)
        return torch.mm(text_norm, proto_norm.t())/temperature

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
        YiJingV53Foundation model ready for inference
    """
    import json
    dir_path = os.path.dirname(os.path.abspath(weights_path))
    config_path = os.path.join(dir_path, "yijing_v53_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f: config = json.load(f)
        vocab_size = config.get("vocab_size", 8145)
    else:
        vocab_size = 8145
    model = YiJingV53Foundation(vocab_size=vocab_size)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def predict(model, text_ids, gua_idx, method='traditional', device='cpu'):
    """
    Run divination prediction.

    Args:
        model: Loaded YiJingV53Foundation model
        text_ids: Tokenized text input, shape (1, seq_len), int tensor
        gua_idx: Hexagram index 0-63
        method: 'traditional' (周易), 'meihua' (梅花), or 'liuyao' (六爻)
        device: 'cpu' or 'cuda'

    Returns:
        dict with prediction outputs
    """
    method_idx = METHOD_MAP.get(method, 0)
    method_tensor = torch.tensor([method_idx], dtype=torch.long, device=device)
    gua_tensor = torch.tensor([gua_idx], dtype=torch.long, device=device)
    symbol_x = torch.tensor([sparse_expand_input(gua_idx)], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(symbol_x, text_ids.to(device), method_tensor, gua_tensor)
    return {k: v.cpu() for k, v in outputs[method].items()}