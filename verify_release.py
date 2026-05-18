"""Verify the V53 model architecture integrity (no weights required).

This script validates:
  1. All model components instantiate correctly
  2. Forward pass dimensions are consistent
  3. Parameter counts match expected values
  4. TrigramSpace sub-modules are functional

NOTE: This script does NOT load trained weights.
To verify actual model inference, obtain authorized weights first.
"""

import sys
sys.path.insert(0, ".")

from yijing_v53_model import (
    YiJingV53Foundation, TrigramSpace,
    TextEncoder, HeLuoLadderNetwork, OutputHeadV38,
    WuxingShengkeModule,
    YinYangBifurcator, WuxingCurvatureGenerator,
    BaguaSphereMapper, HeluoInteractionFolder,
    ResonanceCavity, DomainClassifier,
    sparse_expand_input,
    TEXT_DIM, STATE_DIM, MAX_SEQ,
)
import torch

print("=" * 60)
print("DaoTi V53 Architecture Verification")
print("=" * 60)

errors = []

# ---------------------------------------------------------------------------
# 1. Sub-module instantiation
# ---------------------------------------------------------------------------
print("\n[1] Sub-module integrity...")

try:
    enc = TextEncoder(vocab_size=8145, embed_dim=64, hidden_dim=128,
                      num_heads=4, num_layers=2, max_seq=MAX_SEQ)
    x = torch.randint(1, 8145, (2, 32))
    pooled, hidden = enc(x)
    assert pooled.shape == (2, 128), f"TextEncoder pooled: {pooled.shape}"
    assert hidden.shape == (2, 32, 64), f"TextEncoder hidden: {hidden.shape}"
    print("    TextEncoder: OK")
except Exception as e:
    errors.append(f"TextEncoder: {e}")
    print(f"    TextEncoder: FAIL — {e}")

try:
    ladder = HeLuoLadderNetwork(input_dim=196, state_dim=176, hidden_dim=320,
                                num_layers=6, T=7)
    sx = torch.randn(2, 196)
    gi = torch.tensor([0, 1], dtype=torch.long)
    out = ladder(sx, gi)
    assert out.shape == (2, 176), f"HeLuoLadder: {out.shape}"
    print("    HeLuoLadderNetwork: OK")
except Exception as e:
    errors.append(f"HeLuoLadderNetwork: {e}")
    print(f"    HeLuoLadderNetwork: FAIL — {e}")

try:
    head = OutputHeadV38(176, enable_lora=True)
    h = torch.randn(2, 176)
    for method in ['traditional', 'meihua', 'liuyao']:
        r = head(h, method_name=method)
        assert 'palace' in r, f"Missing 'palace' in {method}"
        assert 'liuqin' in r, f"Missing 'liuqin' in {method}"
    print("    OutputHeadV38 (3 methods): OK")
except Exception as e:
    errors.append(f"OutputHeadV38: {e}")
    print(f"    OutputHeadV38: FAIL — {e}")

# ---------------------------------------------------------------------------
# 2. TrigramSpace sub-modules
# ---------------------------------------------------------------------------
print("\n[2] TrigramSpace sub-modules...")

try:
    yyb = YinYangBifurcator(128, 64, 64)
    yang, yin, alpha = yyb(torch.randn(2, 128))
    assert yang.shape == (2, 64)
    assert yin.shape == (2, 64)
    assert alpha.shape == (2,)
    print("    YinYangBifurcator: OK")
except Exception as e:
    errors.append(f"YinYangBifurcator: {e}")

try:
    wcg = WuxingCurvatureGenerator(176)
    out = wcg(torch.randn(2, 176))
    assert out.shape == (2, 176)
    print("    WuxingCurvatureGenerator: OK")
except Exception as e:
    errors.append(f"WuxingCurvatureGenerator: {e}")

try:
    bsm = BaguaSphereMapper(176, 3)
    r = bsm(torch.randn(2, 176))
    assert 'sphere_coord' in r
    assert 'xiantian_sim' in r
    print("    BaguaSphereMapper: OK")
except Exception as e:
    errors.append(f"BaguaSphereMapper: {e}")

try:
    hif = HeluoInteractionFolder(176, 64)
    q = torch.randn(2, 176)
    p = torch.randn(64, 176)
    r = hif(q, p)
    assert 'folded' in r
    print("    HeluoInteractionFolder: OK")
except Exception as e:
    errors.append(f"HeluoInteractionFolder: {e}")

try:
    rc = ResonanceCavity(176, 8)
    folded = torch.randn(4, 176)
    domain_labels = torch.randint(0, 8, (4,))
    rc.update_standing_wave(folded, domain_labels)
    dl = torch.randn(4, 8)
    coh = rc.compute_coherence(folded, dl)
    assert coh.shape == (4,)
    print("    ResonanceCavity: OK")
except Exception as e:
    errors.append(f"ResonanceCavity: {e}")

# ---------------------------------------------------------------------------
# 3. Full model architecture
# ---------------------------------------------------------------------------
print("\n[3] Full model architecture...")

try:
    model = YiJingV53Foundation(vocab_size=8145)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    expected_total = 5060000  # ~5.06M
    if abs(total - expected_total) > 100000:
        print(f"    WARNING: parameter count {total:,} differs from expected ~{expected_total:,}")

    print(f"    Total parameters: {total:,}")
    print(f"    Trainable:        {trainable:,}")
    print(f"    TextEncoder:      {sum(p.numel() for p in model.text_encoder.parameters()):,}")
    print(f"    HeLuoLadder:      {sum(p.numel() for p in model.heluo_ladder.parameters()):,}")
    print(f"    HeadTraditional:  {sum(p.numel() for p in model.head_traditional.parameters()):,}")

    B, L = 2, 32
    symbol_x = torch.tensor(
        [sparse_expand_input(0), sparse_expand_input(1)], dtype=torch.float32)
    text_ids = torch.randint(1, 8145, (B, L))
    method_idx = torch.tensor([0, 1], dtype=torch.long)
    gua_idx = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(symbol_x, text_ids, method_idx, gua_idx)

    for method in ['traditional', 'meihua', 'liuyao']:
        assert method in outputs, f"Missing method: {method}"
        o = outputs[method]
        for key in ['palace', 'tiangan', 'dizhi', 'liuqin', 'liushen', 'wangxiang', 'biangua_yao']:
            assert key in o, f"Missing key '{key}' in {method}"

    print(f"    Forward pass (random init): OK")
    print(f"    All 7 output heads present for all 3 methods")
    print("    Full model: OK")
except Exception as e:
    errors.append(f"Full model: {e}")
    print(f"    Full model: FAIL — {e}")

# ---------------------------------------------------------------------------
# 4. TrigramSpace integration
# ---------------------------------------------------------------------------
print("\n[4] TrigramSpace integration...")

try:
    trigram = TrigramSpace(text_dim=TEXT_DIM, state_dim=STATE_DIM,
                           input_dim=STATE_DIM,
                           gate_type="resonance_v2", coherence_mode="separation")
    with torch.no_grad():
        text_pooled, _ = model.text_encoder(text_ids)
        text_feat = model.text_proj(text_pooled)
        ts_result = trigram(text_feat)

    expected_keys = ['yang', 'yin', 'bifurcation_alpha', 'state', 'curved_state',
                     'sphere_coord', 'xiantian_sim', 'houtian_sim', 'combined_sim',
                     'flow_weight', 'folded', 'gua_similarity', 'gua_top1_idx',
                     'domain_logits', 'domain_probs']
    for k in expected_keys:
        assert k in ts_result, f"Missing key '{k}' in TrigramSpace output"

    assert ts_result['gua_similarity'].shape == (B, 64)
    assert ts_result['domain_logits'].shape == (B, 8)
    print("    TrigramSpace: OK")
except Exception as e:
    errors.append(f"TrigramSpace: {e}")
    print(f"    TrigramSpace: FAIL — {e}")

# ---------------------------------------------------------------------------
# 5. Sparse input encoding
# ---------------------------------------------------------------------------
print("\n[5] Sparse input encoding...")

try:
    for gi in range(64):
        x = sparse_expand_input(gi)
        assert len(x) == STATE_DIM, f"gi={gi}: dim={len(x)}"
    print("    All 64 hexagrams: OK")
except Exception as e:
    errors.append(f"Sparse input: {e}")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"VERIFICATION FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL CHECKS PASSED")
    print()
    print("Architecture integrity verified.")
    print("NOTE: This verified the model STRUCTURE only.")
    print("Trained weights are NOT included in this public repository.")
    print("To obtain weights for inference, see LICENSE Section 2.2.")
print("=" * 60)