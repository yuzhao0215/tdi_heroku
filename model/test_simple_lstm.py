import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

rnn = nn.LSTM(10, 20, 1)
input = torch.randn(5, 3, 10)

input = input.to(device)
rnn.to(device)

h0 = torch.randn(1, 3, 20).to(device)
c0 = torch.randn(1, 3, 20).to(device)
output, (hn, cn) = rnn(input, (h0, c0))

print(output[-2] == hn)