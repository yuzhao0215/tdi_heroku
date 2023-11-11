import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--------using device: {}---------------".format(device))


# Customized DataLoader class
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, is_test, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

        self.is_test = is_test
        self.previous_data = []  # this is used for test data set. when the index is smaller than sequence_length - 1,
        # previous_data will be used on top of test data for prediction
        if self.is_test:
            self.previous_data = torch.tensor(dataframe[features].values[0:sequence_length - 1]).float()
            self.X = torch.tensor(dataframe[features].values[sequence_length - 1:]).float()
            self.y = torch.tensor(dataframe[target].values[sequence_length - 1:]).float()
        else:
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()
        self.num_features = len(features)

        self.X = self.X.to(device)
        self.y = self.y.to(device)
        if self.is_test:
            self.previous_data = self.previous_data.to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if not self.is_test:
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                if self.num_features > 1:
                    x = self.X[i_start:(i + 1), :]
                else:
                    x = self.X[i_start:(i + 1)]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)

                if self.num_features > 1:
                    x = self.X[0:(i + 1), :]
                else:
                    x = self.X[0:(i + 1)]

                x = torch.cat((padding, x), 0)
        else:
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                if self.num_features > 1:
                    x = self.X[i_start:(i + 1), :]
                else:
                    x = self.X[i_start:(i + 1)]
            else:
                if self.num_features > 1:
                    padding = self.previous_data[i:, :]
                    x = self.X[0:(i + 1), :]
                else:
                    padding = self.previous_data[i:]
                    x = self.X[0:(i + 1)]

                x = torch.cat((padding, x), 0)

        return x, self.y[i]


# class of LSTM
class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)   # ???? should be

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        h0 = h0.to(device)
        c0 = c0.to(device)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])

    if device:
        output = output.to(device)

    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    if device != "cpu":
        return output.cpu()
    else:
        return output