# DaoTi V53 Foundation Model

> **License** · Code: [Apache 2.0](LICENSE_CODE) · Model Weights: [DaoTi Research License v1.0](LICENSE) (no reverse-engineering, no redistribution)
>
> **Core Design**: The DaoTi foundation is permanently frozen. New knowledge is learned exclusively through lightweight adapters. **This project does not provide base model training code, nor does it accept requests to modify base weights.**

**🚀 [Live Demo](https://modelscope.cn/studios/spring30/daoti-v53-spike)** | **📖 [Papers](papers/)** | **📄 [Whitepaper](docs/白皮书_道体基座技术.md)** | **💬 [Discussions](https://github.com/zhibaiYingChuan/DaoTi/issues)**

[中文文档](README.md)

---

## What is DaoTi V53?

DaoTi V53 is a pretrained **semantic foundation model** that takes Chinese natural language text as input and outputs structured semantic representations — including semantic vectors in the encoding space, state vectors in the Luoshu space, and 64-dimensional structured prototype vectors (hexagram space).

Built on the **Bilateral Ladder Network** architecture, trained on general corpora and I Ching classical texts. After training, core parameters are frozen ("DaoTi"), serving as a stable foundation for all domain adaptation. This **Frozen DaoTi + Lightweight Adapter** paradigm is grounded in the discovery of the **Degenerate Ground State** — a gauge-theoretic structure in deep learning. The entire V53 model was trained on a **consumer-grade CPU**, requiring no GPU cluster.

## Key Features

- **Bilateral Ladder Network**: Trained on consumer CPU, no GPU required
- **64-Dimensional Hexagram Space**: Structured reasoning across Eight Palaces, Six Relations, Six Spirits, Heavenly Stems, Earthly Branches, Prosperity-Decline states
- **Frozen Core + Adapter Paradigm**: LoRA domain adapters + physics parameter adapters — core is immutable
- **64-Channel Spike Encoding**: STDP gradient-free learning + architectural safety lock
- **RAG-Enhanced Generation**: Structured reasoning → knowledge base retrieval → natural language response
- **Architectural Safety**: Domain gating / hexagram entropy uncertainty / coherence gating / DaoTi freeze immunity

## Quick Start

```bash
git clone https://github.com/zhibaiYingChuan/DaoTi.git
cd DaoTi
pip install -r requirements.txt
python app.py
```

```python
from inference import load_daoti, predict, generate_response, verify_sha256

verify_sha256("yijing_v53_daoti.pt")
model = load_daoti("yijing_v53_daoti.pt")

from inference import tokenize
text_ids = tokenize("天行健，君子以自强不息")
result = predict(model, text_ids, gua_idx=0, method='traditional')
response = generate_response(model, text_ids, gua_idx=0)
```

## API Reference

See [API_REFERENCE.md](docs/API_REFERENCE.md) for complete function documentation.

## License

| Asset | License | File |
|:---|:---|:---|
| **Code** (inference.py, app.py, train_adapter.py, etc.) | **Apache 2.0** | [LICENSE_CODE](LICENSE_CODE) |
| **Model Weights** (yijing_v53_daoti.pt) | **DaoTi Research License v1.0** | [LICENSE](LICENSE) |

- **Apache 2.0**: Allows commercial use and modification with copyright notice and patent grant.
- **DaoTi Research License v1.0**: Permits inference use, but **prohibits**:
  - Reverse engineering model weights to extract training methodology
  - Using weights to train or distill competing models
  - Redistributing or reselling model weights as a standalone product

Architecture source code and training code are not in this repository and require separate authorization.

## Citation

```bibtex
@software{daoti_v53_2026,
  author = {Independent Researcher, Zhibai},
  title = {DaoTi V53 Foundation: A Semantic Foundation Model Based on Gauge-Theoretic Frozen Architecture},
  year = {2026},
  url = {https://github.com/zhibaiYingChuan/DaoTi}
}
```
