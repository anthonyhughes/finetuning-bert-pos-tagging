import torch
import torch.nn as nn
from transformers import BertModel


class SequenceClassificationModel(nn.Module):
    def __init__(self, vocab_size=None, device=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.fc = nn.Linear(768, vocab_size)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        # training taken from super (default is true)
        if self.training:
            self.bert.train()
            out = self.bert(x)
            encoded_layers = out['last_hidden_state']
        else:
            self.bert.eval()
            with torch.no_grad():
                out = self.bert(x)
                encoded_layers = out['last_hidden_state']

        logits = self.fc(encoded_layers)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
