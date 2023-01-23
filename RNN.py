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
    def __init__(self, input_size=1, hidden_size=50, out_size=13):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]


model = myLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for e in range(100):
    optimizer.zero_grad()
    train = x_train[e].unsqueeze(-1)
    output = model(train)
    loss = criterion(output, y_train[e])

    print(loss)

    loss.backward(retain_graph=True)
    optimizer.step()


























# print(x_train)
#
# rnndata = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# print("Tensor data is: ", rnndata.shape, "\n\n", rnndata)
#
# i_size = 13
# s_length = 1
# h_size = 13
# NUM_LAYERS = 1
# BATCH_SIZE = 1
#
# rnn = nn.RNN(input_size=i_size, hidden_size=h_size, num_layers=1, batch_first=True)
#
# inputs = x_train[0].view(BATCH_SIZE, s_length, i_size)
#
# print(inputs)
#
# out, h_n = rnn(inputs)
#
# print('shape of Input: ', inputs.shape, '\n', inputs)
# print('\n shape of Output: ', out.shape, '\n', out)
# print('\nshape of Hidden: ', h_n.shape, '\n', h_n)
