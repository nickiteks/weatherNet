import pandas as pd
import torch
from torch import nn
import numpy as np
from numpy import genfromtxt

data = pd.read_csv('data.csv', delimiter='\t')
data['time'] = pd.to_datetime(data['time'])

x_train = data[:50000:].drop('time', axis=1)
y_train = data[1:50001].drop('time', axis=1)
x_train = torch.FloatTensor(x_train.values)
y_train = torch.FloatTensor(y_train.values)


# 13 x 100

class optimalNet(nn.Module):
    def __init__(self, n_hid_n):
        super(optimalNet, self).__init__()
        self.fc1 = nn.Linear(13, n_hid_n)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hid_n, 13)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


model = optimalNet(1000)

pred = model.forward(x_train)

optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)


def loss(pred, true):
    sq = (pred - true) ** 2
    return sq.mean()


for e in range(100):
    optimiser.zero_grad()

    y_pred = model.forward(x_train)
    loss_val = loss(y_pred, y_train)

    print(loss_val)

    loss_val.backward()
    optimiser.step()