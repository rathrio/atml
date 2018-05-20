import csv
import torch
from collections import namedtuple


class Metadata:
    Album = namedtuple("Album", ["id", "title", "artist", "artist_id", "songs", "genre", "image_path"])

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

        with open(path_to_metadata, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip headers
            for row in reader:
                self.albums.append(
                    self.Album(
                        id=row[0],
                        title=row[1],
                        artist=row[2],
                        artist_id=row[3],
                        songs=row[4].split("|"),
                        genre=row[5].split("|")[0],
                        image_path=f'/var/tmp/albums/data/{row[0]}/{row[3]}/primary.jpg'
                    )
                )

        print(f'Loaded {len(self.albums)} albums')

    def genre_index(self, genre):
        return self.genres.index(genre)

    def genre(self, index):
        return self.genres[index]

    def album(self, index):
        return self.albums[index]


