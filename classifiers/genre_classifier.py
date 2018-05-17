import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import csv
import IPython

class Metadata:
    genres = ["Electronic",
              "Rock",
              "Pop",
              "Jazz",
              "Hip Hop",
              "Funk / Soul",
              "Folk, World, & Country",
              "Classical",
              "Reggae",
              "Non-Music",
              "Latin",
              "Blues",
              "Children's",
              "Stage & Screen",
              "Brass & Military"]

    genre_count = len(genres)

    def __init__(self, path_to_metadata):
        self.path_to_metadata = path_to_metadata
        self.albums = []

        with open(path_to_metadata) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip headers
            for row in reader:
                self.albums.append(row)

        print(f'Loaded {len(self.albums)} albums')

    def genre_tensor(self, genre):
        index = self.genres.index(genre)
        t = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        t[index] = 1
        return t

    def album(self, index):
        return self.albums[index]

    def genre(self, index):
        return self.album(index)[5].split("|")[0]


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

IPython.embed()


for epoch in range(2):
    pass
