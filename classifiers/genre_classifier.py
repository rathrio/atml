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
save_model_path = "./data/models"

cuda_available = torch.cuda.is_available()


class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, genre_count):
        super(GenreClassifier, self).__init__()
        self.genre_count = genre_count
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, genre_count, bias=False)

    def forward(self, features):
        out = self.fc1(features)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


def train():
    metadata = Metadata(metadata_path)
    albums = metadata.albums
    all_features = np.load(features_path)
    print(f'Loaded {len(all_features)} features')

    assert len(albums) == len(all_features)

    input_size = 4096
    hidden_size = 1000
    genre_count = metadata.genre_count
    net = GenreClassifier(input_size, hidden_size, genre_count)
    if cuda_available:
        net = net.cuda()

    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    batch_size = 1000

    # try:
    for epoch in range(10):

        running_loss = 0.0
        for i, album in enumerate(albums, 0):
            features = torch.Tensor([all_features[i].tolist()])
            if cuda_available:
                features = features.cuda()
            features = Variable(features)

            genre = torch.LongTensor([metadata.genre_index(album.genre)])
            if cuda_available:
                genre = genre.cuda()
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
    # except:
    #     torch.save(net.state_dict(), f'{save_model_path}/genre_model')

    torch.save(net.state_dict(), f'{save_model_path}/genre_classifier.pkl')
    print('Done!')


# metadata = Metadata(metadata_path)
# albums = metadata.albums
# all_features = np.load(features_path)
# print(f'Loaded {len(all_features)} features')

# assert len(albums) == len(all_features)

# input_size = 4096
# hidden_size = 1000
# genre_count = metadata.genre_count
# net = GenreClassifier(input_size, hidden_size, genre_count)
# net.load_state_dict(torch.load('./classifiers/genre_classifier.pkl'))

# IPython.embed()


train()
