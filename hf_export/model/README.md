---
license: mit
language:
- code
base_model: microsoft/codebert-base
pipeline_tag: text-classification
tags:
- flaky-tests
- software-testing
- code
- codebert
- reproduction
---

# CodeBERT for Flaky Test Categorisation (FlakeBench, fold 2)

Classifies a Java/Kotlin test method into one of six categories: five kinds of flaky
test plus non-flaky.

## What this is

A fine-tune of `microsoft/codebert-base` on the FlakeBench dataset from
[*Understanding and Improving Flaky Test Classification*](https://utexas.app.box.com/v/august-shi-OOPSLA2025)
(OOPSLA 2025), trained as a reproduction exercise on a single 8 GB consumer GPU.

The uploaded weights are the "Balanced" configuration below.

## Training configurations

| Parameter | Baseline | lr 2e-5 | Balanced | Augmented | Paper |
| --- | --- | --- | --- | --- | --- |
| Encoder | codebert-base | codebert-base | codebert-base | codebert-base | codebert-base |
| Learning rate | 1e-5 | 2e-5 | 1e-5 | 1e-5 | 1e-5 |
| Batch size | 8 | 8 | 8 | 8 | 8 |
| Max length | 512 | 512 | 512 | 512 | 512 |
| Loss | focal γ=2.0 | focal γ=2.0 | focal γ=2.0 | focal γ=2.0 | focal γ=2.0 |
| Class weights | balanced | balanced | balanced | balanced | balanced |
| Optimizer | AdamW wd 0.01 | AdamW wd 0.01 | AdamW wd 0.01 | AdamW wd 0.01 | AdamW wd 0.01 |
| Precision | fp16 | fp16 | fp16 | fp16 | fp32 |
| Non-flaky rows | 4,972 | 4,972 | 800 | 800 | full |
| Minority handling | none | none | ×160 copies | ×200 variants | none |
| Train rows | 5,114 | 5,114 | 1,600 | 1,800 | 5,114 |
| Epochs run | 8 | 8 | 18 | 13 | 40 |
| Dynamic padding | no | no | no | no | no |

Hardware: 1× RTX 4060 Laptop (8 GB). Class rebalancing is the one deviation from the
paper's method, which trains on the raw distribution (97% non-flaky).

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

name = "Ariful1904129/codebert-flakytest-fold2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True).eval()

code = """@Test
public void testConnect() throws Exception {
    Thread.sleep(1000);
    assertTrue(client.isConnected());
}"""

x = tok(code, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    pred = model(**x).logits.argmax(-1).item()
print(model.config.id2label[pred])
```

`trust_remote_code=True` is required — the MLP head is a custom architecture defined in
`modeling_flakylens.py`.

Labels: `0` async wait, `1` concurrency, `2` time, `3` unordered collections,
`4` test order dependency, `5` non-flaky.

Scope: Java/Kotlin test methods; inputs longer than 512 tokens are truncated.

## Citation

Please cite the original paper. This model is a third-party reproduction and is not
endorsed by its authors.

```bibtex
@inproceedings{flakylens2025,
  title     = {Understanding and Improving Flaky Test Classification},
  booktitle = {OOPSLA},
  year      = {2025}
}
```

Dataset and method: [UT-SE-Research/FlakyLens](https://github.com/UT-SE-Research/FlakyLens).
