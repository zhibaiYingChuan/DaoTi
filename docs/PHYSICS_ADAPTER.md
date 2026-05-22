# DaoTi V53 物理适配器训练数据格式

## JSONL 格式

每行一条 JSON 记录，包含 `params`（物理参数）和 `spectrum`（光谱值）字段：

```jsonl
{"params": [0.5, 0.3, 1.45, 0.2, 25.0, 0.01, 0.0, 0.0], "spectrum": [0.01, 0.02, 0.05, ..., 0.01]}
{"params": [0.6, 0.4, 1.50, 0.3, 30.0, 0.02, 0.0, 0.0], "spectrum": [0.02, 0.03, 0.08, ..., 0.02]}
```

## 字段说明

| 字段 | 类型 | 必需 | 说明 |
|:-----|:-----|:-----|:-----|
| `params` | float[] | ✅ | 物理结构参数，长度必须等于 `--input_dim` |
| `spectrum` | float[] | ✅ | 目标光谱值，长度必须等于 `--output_dim` |

## 典型光学参数示例

| 参数索引 | 典型含义 | 单位 | 典型范围 |
|:---------|:---------|:-----|:---------|
| 0 | 光栅周期 | μm | 0.3 - 2.0 |
| 1 | 占空比 | — | 0.1 - 0.9 |
| 2 | 折射率 | — | 1.0 - 3.5 |
| 3 | 厚度 | μm | 0.01 - 10.0 |
| 4-7 | 其他参数 | — | 视具体结构 |

> 以上仅为示例。`params` 的含义完全由用户定义，适配器不预设任何物理假设。

## 训练命令

```bash
python scripts/train_physics_adapter.py \
    --data_path ./optics_data.jsonl \
    --output_path ./optics_adapter.pt \
    --input_dim 8 \
    --output_dim 100 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3
```

## 推理命令

```python
from daoti.inference import load_daoti, load_physics_adapter, predict_physics
import torch

model = load_daoti("weights/yijing_v53_daoti.pt")
adapter_data = load_physics_adapter("optics_adapter.pt")

params = torch.tensor([[0.5, 0.3, 1.45, 0.2, 25.0, 0.01, 0.0, 0.0]])
result = predict_physics(model, adapter_data, params)
print(result['spectrum'])  # 预测的光谱值
```

## 数据量建议

| 规模 | 记录数 | 效果 |
|:-----|:-------|:-----|
| 最小验证 | 20+ | 验证流程可运行 |
| 基础预测 | 100-500 | 适配器开始学习物理映射 |
| 精确预测 | 1000+ | 适配器收敛，R² > 0.9 |

## 注意事项

- 数据自动进行零均值单位方差归一化，归一化参数保存在适配器文件中
- 推理时自动反归一化，输出为原始量纲的光谱值
- 道体核心权重在训练过程中始终冻结
- `--input_dim` 和 `--output_dim` 必须与数据文件中的实际维度一致
- 验证集比例默认 20%，可通过 `--val_split` 调整
