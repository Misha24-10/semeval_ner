from torch import nn
import torch

class Ensemble_model(torch.nn.Module):

    def __init__(self, num_models=3):
        super(Ensemble_model, self).__init__()
        self.num_labels = 73
        self.hidden = 256
        self.linear_1 = torch.nn.Linear(self.num_labels * num_models, self.hidden)
        self.func = nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden, 73)

    def forward(self, x, labels=None):
        logits = self.linear_2(self.func(self.linear_1(x)))
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits