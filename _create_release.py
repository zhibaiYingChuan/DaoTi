import json
import urllib.request
import urllib.error
import sys

REPO = "zhibaiYingChuan/DaoTi"
TAG = "v1.0.0"

body_text = """## DaoTi V53 Foundation Model — 首个公开发布版

### 核心特性

- **双轨阶梯网络架构**：消费级CPU训练完成，无需GPU集群
- **64维卦象空间结构化推理**：八宫/六亲/六神/天干地支/旺相休囚/五行
- **冻结道体 + 轻量适配器范式**：LoRA领域适配 + 物理参数适配，核心不可篡改
- **64通道脉冲编码接口**：STDP无梯度学习 + 架构级安全锁机制
- **RAG检索增强生成**：结构化推理结果 → 知识库检索 → 自然语言回答
- **架构级安全机制**：领域门控 / 卦象熵不确定性度量 / 相干性门控 / 道体冻结免疫

### 防御性保护

- 双许可证摘要醒目展示（Apache 2.0 + DaoTi Research License v1.0）
- 核心设计声明：道体永久冻结，不提供基座训练代码
- 超参数常量抽象，避免训练工艺直接暴露
- 代码审计通过，无训练逻辑泄露

### 包含文件

| 文件 | 说明 |
|:---|:---|
| `yijing_v53_daoti.pt` | 模型权重 (state_dict) |
| `inference.py` | 推理接口 (加载 + 预测 + 适配器) |
| `_model_core.py` | 架构定义 (推理构建版) |
| `_constants.py` | 数据常量 |
| `app.py` | Gradio 交互式演示 |
| `train_adapter.py` | 领域适配器训练脚本 |
| `train_physics_adapter.py` | 物理参数适配器训练脚本 |
| `spike_interface.py` | 64通道脉冲编码接口 |
| `papers/` | 6 篇研究论文 |

### 双许可证

- **代码**: Apache 2.0 — 允许商业使用和修改
- **模型权重**: DaoTi Research License v1.0 — 禁止逆向工程、禁止再分发

### 快速开始

```bash
git clone https://github.com/zhibaiYingChuan/DaoTi.git
cd DaoTi
pip install -r requirements.txt
python app.py
```

### 在线演示

- [ModelScope 魔搭空间](https://modelscope.cn/studios/spring30/daoti-v53-spike) — 64通道脉冲编码接口交互式演示
"""

payload = json.dumps({
    "tag_name": TAG,
    "name": f"DaoTi V53 Foundation Model {TAG}",
    "body": body_text,
    "draft": False,
    "prerelease": False,
}).encode("utf-8")

url = f"https://api.github.com/repos/{REPO}/releases"
req = urllib.request.Request(url, data=payload, method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("Accept", "application/vnd.github+json")

token = sys.argv[1] if len(sys.argv) > 1 else None
if token:
    req.add_header("Authorization", f"Bearer {token}")

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        print(f"Release created: {result.get('html_url', 'OK')}")
except urllib.error.HTTPError as e:
    err = e.read().decode("utf-8")
    print(f"HTTP {e.code}: {err}")
except Exception as e:
    print(f"Error: {e}")
