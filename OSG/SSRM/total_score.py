import torch
import torch.nn as nn
import numpy as np

class TTC(nn.Module):
    def __init__(self):
        super(TTC, self).__init__()
## 4 layers
        self.fc1 = nn.Linear(10240, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        self.activate = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.activate(x)


        return x