import csv
import torch


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