import torch
import torch.nn as nn
import torch.optim as optim


class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, genre_count):
        super(GenreClassifier, self).__init__()
        self.genre_count = genre_count
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, genre_count, bias=False)

    def forward(self, features):
        out = self.fc1(features)
        out = self.fc2(out)
        return nn.Softmax(out)


net = GenreClassifier(4096, 1000, 15)
learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(2):
    pass
