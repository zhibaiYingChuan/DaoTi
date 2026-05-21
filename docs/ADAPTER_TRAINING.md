# DaoTi V53 适配器训练数据格式

## JSONL 格式

每行一条 JSON 记录，包含 `text` 字段和可选的 `domain` 字段：

```jsonl
{"text": "光栅周期为500纳米时，共振波长出现在1550纳米附近", "domain": "optics"}
{"text": "当占空比从0.3增加到0.7，Q因子先升后降", "domain": "optics"}
{"text": "此卦利于远行，不利近求", "domain": "divination"}
```

## 字段说明

| 字段 | 类型 | 必需 | 说明 |
|:-----|:-----|:-----|:-----|
| `text` | string | ✅ | 训练文本，至少2个字符 |
| `domain` | string | ❌ | 领域标签，不提供时使用 `--domain_name` 参数值 |

## 数据量建议

| 规模 | 记录数 | 效果 |
|:-----|:-------|:-----|
| 最小验证 | 50+ | 验证流程可运行 |
| 基础适配 | 200-500 | 适配器开始学习领域特征 |
| 充分适配 | 1000+ | 适配器稳定收敛 |

## 训练命令

```bash
python train_adapter.py \
    --data_path ./my_data.jsonl \
    --output_path ./my_adapter.pt \
    --domain_name "optics" \
    --method traditional \
    --epochs 10 \
    --lr 1e-4
```

## 加载适配器推理

```python
from inference import load_daoti, load_adapter, predict

model = load_daoti("yijing_v53_daoti.pt")
model = load_adapter(model, "my_adapter.pt")

result = predict(model, text_ids, gua_idx=0, method='traditional')
```

## 注意事项

- 道体核心权重在训练过程中始终冻结，适配器只修改 LoRA 参数
- 训练需要 `daoti_v53_tokenizer.pt` 分词器文件
- 中文文本效果最佳，英文文本需要额外处理
- `domain` 字段用于构建训练目标，相同领域的文本应使用相同的 domain 值
