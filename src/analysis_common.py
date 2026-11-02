"""Shared loader for the failure-analysis steps: rebuilds the repo's BERT_Arch from the
weights published on HuggingFace (the only surviving checkpoint) so the repo's own
attribution (cig.py) and perturbation (perturbation.py) code can be reused unchanged."""
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from codebert_model import BERT_Arch
from utils import codebert_model_define

NAMES = ["Async Wait", "Concurrency", "Time", "Unordered Collections",
         "Test Order Dependency", "Non-flaky"]
DEV = torch.device("cuda")


def load_model():
    _, tokenizer, auto_model = codebert_model_define()
    model = BERT_Arch(auto_model, 6)
    sd = load_file("../models/hf_model.safetensors")
    # HF export renamed the encoder prefix bert. -> roberta.; map it back
    sd = {("bert." + k[len("roberta."):] if k.startswith("roberta.") else k): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("position_ids" in m for m in missing), missing
    return tokenizer, model.to(DEV).eval()


def load_test(fold=2):
    return pd.read_csv(f"FlakyLens_Categorization_PerProject-Data/test_set_{fold}.csv")


@torch.no_grad()
def predict(model, tokenizer, codes, batch=16):
    """Returns (pred_ids, probs[n,6])."""
    enc = tokenizer.batch_encode_plus(list(codes), max_length=512,
                                      padding="max_length", truncation=True)
    seq = torch.tensor(enc["input_ids"]); msk = torch.tensor(enc["attention_mask"])
    out = []
    for i in range(0, len(seq), batch):
        lp = model(seq[i:i + batch].to(DEV), msk[i:i + batch].to(DEV))  # log-probs
        out.append(lp.float().cpu())
    lp = torch.cat(out)
    return lp.argmax(1).numpy(), lp.exp().numpy()
