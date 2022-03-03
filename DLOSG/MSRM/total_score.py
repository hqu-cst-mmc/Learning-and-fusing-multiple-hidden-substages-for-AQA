import torch
import torch.nn as nn
import numpy as np

class TTC(nn.Module):
    def __init__(self):
        super(TTC, self).__init__()
        self.fc_ttc = nn.Linear(5,1)
        self.activate = nn.Sigmoid()
    def forward(self, x):
        x = self.fc_ttc(x)
        x = self.activate(x)
        return x