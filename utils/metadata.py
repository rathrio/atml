#!/usr/bin/env python3

import csv
import numpy as np


class Metadata:
    def __init__(self, path_to_metadata):
        self.path_to_metadata = path_to_metadata
        self.albums = []

        with open(path_to_metadata) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip headers
            for row in reader:
                self.albums.append(row)

        print(f'Loaded {len(self.albums)} albums')

    def genre_vector:
        pass

    def album(self, index):
        return self.albums[index]

    def genre(self, index):
        return self.album(index)[5].split("|")[0]
