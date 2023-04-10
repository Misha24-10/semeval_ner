from torch import nn
import torch

class Ensemble_model(torch.nn.Module):

    def __init__(self, hidden):
        super(Ensemble_model, self).__init__()
        self.num_labels = 73
        self.hidden = hidden
        self.linear_1 = torch.nn.Linear(self.num_labels * 3, self.hidden)
        self.func = nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear_3 = torch.nn.Linear(self.hidden, 73)

    def forward(self, x, labels=None):
        logits = self.func(self.linear_1(x))
        logits = self.func(self.linear_2(logits))
        logits = self.linear_3(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
