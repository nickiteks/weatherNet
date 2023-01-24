import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch  # pytorch
import torch.nn as nn
from torch.autograd import Variable

data = pd.read_csv('data.csv', delimiter='\t', index_col='time', parse_dates=True)

x_train = data[:10000:]
y_train = data[1:10001:]

print(data.head(5))

print("x", x_train)
print(y_train)

mm = MinMaxScaler()
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
y_train = mm.fit_transform(y_train)

x_train = Variable(torch.Tensor(x_train))
y_train = Variable(torch.Tensor(y_train))

x_train = torch.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


num_epochs = 1000  # 1000 epochs
learning_rate = 0.001  # 0.001 lr

input_size = 13  # number of features
hidden_size = 2  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers

num_classes = 13  # number of output classes

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train.shape[1])  # our lstm class

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm1.forward(x_train)  # forward pass
    optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = criterion(outputs, y_train)

    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, i.e backprop
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

x = data[10000:10001:]
print(x)
x = Variable(torch.Tensor(ss.fit_transform(x)))
x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))

print(x)

pred = lstm1.forward(x)

print(mm.inverse_transform(pred.data.numpy()))