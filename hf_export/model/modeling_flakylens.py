"""CodeBERT + MLP head for flaky-test category classification.

Mirrors BERT_Arch from the FlakyLens artifact: RoBERTa pooled output -> Linear(768,512)
-> ReLU -> Dropout -> Linear(512,6). The original applies LogSoftmax to the final layer;
this returns raw logits instead, following the HF convention. argmax is identical either
way, so predictions match exactly -- apply log_softmax if you need the original's values.
"""
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FlakyLensConfig(RobertaConfig):
    model_type = "flakylens"

    def __init__(self, head_hidden=512, head_dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout


class FlakyLensForTestClassification(RobertaPreTrainedModel):
    config_class = FlakyLensConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.fc1 = nn.Linear(config.hidden_size, config.head_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.head_dropout)
        self.fc2 = nn.Linear(config.head_hidden, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]
        logits = self.fc2(self.dropout(self.relu(self.fc1(pooled))))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
