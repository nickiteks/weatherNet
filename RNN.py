import numpy as np
import pandas as pd
import torch
import torchvision.datasets
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import transforms

data = pd.read_csv('data.csv', delimiter='\t')

x_train = data[:50000:].drop('time', axis=1)
y_train = data[1:50001].drop('time', axis=1)
x_train = torch.FloatTensor(x_train.values)
y_train = torch.FloatTensor(y_train.values)


class myLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, out_size=13):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        lstm_out, hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]


model = myLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for e in range(10000):
    optimizer.zero_grad()
    train = x_train[e].unsqueeze(-1)

    output = model(train)
    loss = criterion(output, y_train[e])

    print(loss)

    loss.backward()
    optimizer.step()
