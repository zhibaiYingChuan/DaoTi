import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_daoti, STATE_DIM


class DaoTiInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, symbol_x, text_ids, method_idx, gua_idx):
        outputs = self.model(symbol_x, text_ids, method_idx, gua_idx)
        trad = outputs['traditional']
        meihua = outputs['meihua']
        liuyao = outputs['liuyao']

        result = torch.cat([
            trad['palace'],
            trad['tiangan'],
            trad['dizhi'],
            trad['liuqin'],
            trad['liushen'],
            trad['wangxiang'],
            trad['biangua_yao'],
            trad['palace_wuxing'],
            trad['dizhi_wuxing'],
            meihua['palace'],
            meihua['tiangan'],
            meihua['dizhi'],
            meihua['liuqin'],
            meihua['liushen'],
            meihua['wangxiang'],
            meihua['biangua_yao'],
            meihua['palace_wuxing'],
            meihua['dizhi_wuxing'],
            liuyao['palace'],
            liuyao['tiangan'],
            liuyao['dizhi'],
            liuyao['liuqin'],
            liuyao['liushen'],
            liuyao['wangxiang'],
            liuyao['biangua_yao'],
            liuyao['palace_wuxing'],
            liuyao['dizhi_wuxing'],
            outputs['text_feat'],
        ], dim=-1)

        return result

    def encode_text_coherence(self, text_ids, gua_idx):
        text_feat = self.model.encode_text(text_ids)
        proto = self.model.gua_prototype.weight
        import torch.nn.functional as F
        proto_n = F.normalize(proto, p=2, dim=-1)
        feat_n = F.normalize(text_feat, p=2, dim=-1)
        similarity = torch.mm(feat_n, proto_n.t()).squeeze()
        top_sim = similarity.max().item()
        coherence = max(0.0, min(1.0, top_sim))
        return torch.tensor([coherence])


OUTPUT_SLICES = {
    'traditional': {
        'palace': (0, 8),
        'tiangan': (8, 18),
        'dizhi': (18, 30),
        'liuqin': (30, 36),
        'liushen': (36, 42),
        'wangxiang': (42, 47),
        'biangua_yao': (47, 53),
        'palace_wuxing': (53, 58),
        'dizhi_wuxing': (58, 63),
    },
    'meihua': {
        'palace': (63, 71),
        'tiangan': (71, 81),
        'dizhi': (81, 93),
        'liuqin': (93, 99),
        'liushen': (99, 105),
        'wangxiang': (105, 110),
        'biangua_yao': (110, 116),
        'palace_wuxing': (116, 121),
        'dizhi_wuxing': (121, 126),
    },
    'liuyao': {
        'palace': (126, 134),
        'tiangan': (134, 144),
        'dizhi': (144, 156),
        'liuqin': (156, 162),
        'liushen': (162, 168),
        'wangxiang': (168, 173),
        'biangua_yao': (173, 179),
        'palace_wuxing': (179, 184),
        'dizhi_wuxing': (184, 189),
    },
    'text_feat': (189, 365),
}


def parse_flat_output(flat_tensor, method='traditional'):
    result = {}
    for key, (start, end) in OUTPUT_SLICES[method].items():
        result[key] = flat_tensor[:, start:end]
    result['text_feat'] = flat_tensor[:, 189:365]
    return result


print("Loading model...")
model = load_daoti("yijing_v53_daoti.pt", device="cpu")
model.eval()

wrapper = DaoTiInferenceWrapper(model)
wrapper.eval()

symbol_x = torch.randn(1, STATE_DIM)
text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long)
method_idx = torch.tensor([0], dtype=torch.long)
gua_idx = torch.tensor([0], dtype=torch.long)

print("Testing wrapper...")
with torch.no_grad():
    flat = wrapper(symbol_x, text_ids, method_idx, gua_idx)
print(f"Wrapper output shape: {flat.shape}")

parsed = parse_flat_output(flat, 'traditional')
print(f"Parsed keys: {list(parsed.keys())}")
print(f"Palace: {parsed['palace'].argmax().item()}")

print("\nTrying TorchScript trace on wrapper...")
try:
    traced = torch.jit.trace(wrapper, (symbol_x, text_ids, method_idx, gua_idx))
    traced.save("daoti_traced.pt")
    print("TorchScript trace SUCCESS!")

    loaded = torch.jit.load("daoti_traced.pt")
    with torch.no_grad():
        flat2 = loaded(symbol_x, text_ids, method_idx, gua_idx)
    print(f"Traced output shape: {flat2.shape}")

    diff = (flat - flat2).abs().max().item()
    print(f"Max diff: {diff:.8f}")

    parsed2 = parse_flat_output(flat2, 'traditional')
    print(f"Traced Palace: {parsed2['palace'].argmax().item()}")

    size_mb = os.path.getsize("daoti_traced.pt") / 1024 / 1024
    print(f"Traced model size: {size_mb:.1f} MB")
    print(f"\n✅ SUCCESS: daoti_traced.pt can be loaded WITHOUT _model_core.py!")

except Exception as e:
    print(f"TorchScript trace FAILED: {e}")
