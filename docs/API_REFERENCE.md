# DaoTi V53 API Reference

> **License** · Code: [Apache 2.0](LICENSE_CODE) · Model Weights: [DaoTi Research License v1.0](LICENSE)

## Quick Start

```python
from inference import load_daoti, predict, generate_response, verify_sha256

verify_sha256("yijing_v53_daoti.pt")
model = load_daoti("yijing_v53_daoti.pt")

from inference import tokenize
text_ids = tokenize("天行健，君子以自强不息", max_seq=256)
result = predict(model, text_ids, gua_idx=0, method='traditional')
response = generate_response(model, text_ids, gua_idx=0, method='traditional')
```

---

## Core Functions

### `verify_sha256(weights_path, hash_path=None)`

Verify model weight file integrity via SHA256 checksum.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `weights_path` | `str` | required | Path to `.pt` weight file |
| `hash_path` | `str \| None` | `None` | Path to `.sha256` file; defaults to `weights_path + ".sha256"` |

**Returns:** `bool` — `True` if hash matches, `False` otherwise.

---

### `load_daoti(weights_path, device='cpu')`

Load the DaoTi V53 foundation model from a `.pt` weights file. Automatically reads `vocab_size` from `yijing_v53_config.json` in the same directory.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `weights_path` | `str` | required | Path to `yijing_v53_daoti.pt` |
| `device` | `str` | `'cpu'` | `'cpu'` or `'cuda'` |

**Returns:** `nn.Module` — Model in eval mode, ready for inference.

---

### `predict(model, text_ids, gua_idx, method='traditional', device='cpu')`

Execute structured reasoning on the model.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model` | `nn.Module` | required | Loaded model from `load_daoti()` |
| `text_ids` | `Tensor` | required | Tokenized input, shape `(1, seq_len)`, dtype `torch.long` |
| `gua_idx` | `int` | required | Hexagram index, 0–63 |
| `method` | `str` | `'traditional'` | Reasoning method: `'traditional'`, `'meihua'`, or `'liuyao'` |
| `device` | `str` | `'cpu'` | `'cpu'` or `'cuda'` |

**Returns:** `dict` with the following keys:

| Key | Shape | Description |
|:---|:---|:---|
| `palace` | `(1, 8)` | Eight-palace classification logits |
| `tiangan` | `(1, 10)` | Heavenly Stem logits |
| `dizhi` | `(1, 12)` | Earthly Branch logits |
| `liuqin` | `(1, 6)` | Six Relations logits |
| `liushen` | `(1, 6)` | Six Spirits logits |
| `wangxiang` | `(1, 5)` | Prosperity-decline logits |
| `biangua_yao` | `(1, 6)` | Moving line raw scores |
| `palace_wuxing` | `(1, 5)` | Palace Five-Element logits |
| `dizhi_wuxing` | `(1, 5)` | Branch Five-Element logits |
| `coherence` | `float` | Self-calibrating quality signal in [0, 1] |

---

### `compute_coherence(model, text_ids, gua_idx, device='cpu')`

Compute resonance cavity coherence — a self-calibrating quality signal. Higher values indicate the model is more confident.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model` | `nn.Module` | required | Loaded model |
| `text_ids` | `Tensor` | required | Tokenized input, shape `(1, seq_len)` |
| `gua_idx` | `int` | required | Hexagram index, 0–63 |
| `device` | `str` | `'cpu'` | Device |

**Returns:** `float` — Coherence score in [0, 1].

---

### `generate_response(model, text_ids, gua_idx, method='traditional', device='cpu', coherence_threshold=0.3)`

Generate a structured natural language response via retrieval-augmented generation (RAG). Automatically appends a low-confidence warning if coherence falls below threshold.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model` | `nn.Module` | required | Loaded model |
| `text_ids` | `Tensor` | required | Tokenized input, shape `(1, seq_len)` |
| `gua_idx` | `int` | required | Hexagram index, 0–63 |
| `method` | `str` | `'traditional'` | Reasoning method |
| `device` | `str` | `'cpu'` | Device |
| `coherence_threshold` | `float` | `0.3` | Below this value, a warning is appended |

**Returns:** `dict` with:

| Key | Type | Description |
|:---|:---|:---|
| `response` | `str` | Composed natural language response |
| `coherence` | `float` | Quality signal |
| `low_confidence` | `bool` | Whether coherence < threshold |
| `details` | `dict` | Structured breakdown (gua_name, palace, tiangan, dizhi, liuqin, liushen, wangxiang, moving_yao, coherence) |

---

## Adapter Functions

### `load_adapter(model, adapter_path, device='cpu')`

Load a trained domain LoRA adapter into the model. The adapter modifies only the task head weights; the DaoTi core remains frozen.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model` | `nn.Module` | required | Loaded model |
| `adapter_path` | `str` | required | Path to adapter `.pt` file |
| `device` | `str` | `'cpu'` | Device |

**Returns:** `nn.Module` — Model with adapter weights applied.

---

### `load_physics_adapter(adapter_path, device='cpu')`

Load a physics parameter adapter for spectrum prediction. Returns adapter data dict (not a model).

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `adapter_path` | `str` | required | Path to physics adapter `.pt` file |
| `device` | `str` | `'cpu'` | Device |

**Returns:** `dict | None` — Adapter data with keys: `weights`, `norm_stats`, `input_dim`, `output_dim`, `type`.

---

### `predict_physics(daoti_model, adapter_data, physics_params, device='cpu')`

Predict spectrum from physics parameters using a physics adapter.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `daoti_model` | `nn.Module` | required | Loaded DaoTi model |
| `adapter_data` | `dict` | required | From `load_physics_adapter()` |
| `physics_params` | `Tensor` | required | Shape `(batch, input_dim)` or `(input_dim,)` |
| `device` | `str` | `'cpu'` | Device |

**Returns:** `dict` with:

| Key | Type | Description |
|:---|:---|:---|
| `spectrum` | `Tensor` | Denormalized predicted spectrum |
| `spectrum_norm` | `Tensor` | Normalized predicted spectrum |

---

## Utility Functions

### `sparse_expand_input(gua_idx)`

Expand hexagram index into a 176-dimensional sparse state vector using the trigram encoding scheme.

| Parameter | Type | Description |
|:---|:---|:---|
| `gua_idx` | `int` | Hexagram index, 0–63 |

**Returns:** `list[float]` — 176-dim state vector.

### `find_palace(gua_idx)`

Find the Eight-Palace归属 for a given hexagram.

| Parameter | Type | Description |
|:---|:---|:---|
| `gua_idx` | `int` | Hexagram index, 0–63 |

**Returns:** `int` — Palace index, 0–7.

### `tokenize(text, max_seq=256)`

Tokenize Chinese text using the built-in tokenizer mapping. Returns a padded tensor ready for model input.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `text` | `str` | required | Chinese text input |
| `max_seq` | `int` | `256` | Maximum sequence length |

**Returns:** `Tensor` — Shape `(1, max_seq)`, dtype `torch.long`.

---

## Constants

| Name | Type | Description |
|:---|:---|:---|
| `GUA_64` | `list[str]` | Names of all 64 hexagrams |
| `BA_GONG` | `list[list[int]]` | Eight-Palace hexagram groupings |
| `GUA_WUXING` | `list[str]` | Five-Element assignment per hexagram |
| `GUA_TRIGRAM` | `dict` | Trigram decomposition |
| `METHOD_MAP` | `dict` | `{'traditional': 0, 'meihua': 1, 'liuyao': 2}` |
| `STATE_DIM` | `int` | 176 — Core state dimension |
| `TEXT_DIM` | `int` | 128 — Text encoder output dimension |
| `MAX_SEQ` | `int` | 256 — Maximum sequence length |

---

## Error Handling

| Scenario | Behavior |
|:---|:---|
| Weight file not found | `torch.load` raises `FileNotFoundError` |
| SHA256 mismatch | `verify_sha256()` returns `False`, prints warning |
| Invalid gua_idx | Model processes but output is undefined for indices outside 0–63 |
| Invalid method | Falls back to `'traditional'` via `METHOD_MAP.get(method, 0)` |
| Missing config.json | Uses default `vocab_size=8145` |
| Non-physics adapter file | `load_physics_adapter()` returns `None` with warning |
