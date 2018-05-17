import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from metadata import Metadata

import IPython


class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, genre_count):
        super(GenreClassifier, self).__init__()
        self.genre_count = genre_count
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, genre_count, bias=False)
        self.sm = nn.Softmax()

    def forward(self, features):
        out = self.fc1(features)
        out = self.fc2(out)
        return self.sm(out)


metadata = Metadata('./data/metadata.csv')

input_size = 4096
hidden_size = 1000
genre_count = 15
net = GenreClassifier(input_size, hidden_size, genre_count)
learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

features = Variable(torch.randn(input_size))
out = net(features)


for epoch in range(2):
    pass
