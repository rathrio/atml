import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from metadata import Metadata

import IPython

metadata_path = "./data/metadata.csv"
features_path = "./data/music_alb.npy"


class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, genre_count):
        super(GenreClassifier, self).__init__()
        self.genre_count = genre_count
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, genre_count, bias=False)

    def forward(self, features):
        out = self.fc1(features)
        out = self.fc2(out)
        return F.softmax(out)


def train():
    metadata = Metadata(metadata_path)
    albums = metadata.albums
    all_features = np.load(features_path)
    print(f'Loaded {len(all_features)} features')

    assert len(albums) == len(all_features)

    input_size = 4096
    hidden_size = 512
    genre_count = metadata.genre_count
    net = GenreClassifier(input_size, hidden_size, genre_count)

    learning_rate = 1e-3
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    batch_size = 100

    for epoch in range(2):

        running_loss = 0.0
        for i, album in enumerate(albums, 0):
            features = torch.from_numpy(all_features[i])
            features = Variable(features)
            genre = metadata.genre_tensor(album.genre)
            genre = Variable(genre)

            optimizer.zero_grad()

            out = net(features)
            loss = criterion(out, genre)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            running_loss += loss.data[0]
            if i % batch_size == (batch_size - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / batch_size))
                running_loss = 0.0

    print('Done!')


train()
