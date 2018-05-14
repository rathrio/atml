#!/usr/bin/env python3

import os
import sys
import csv

with open("../data/metadata.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader, None) # skip headers

    for row in reader:
      album_id = row[0]
      title = row[1]
      artist_id = row[3]
      genres = row[5]
      main_genre = genres.split("|")[0]
      image_path = f'/var/tmp/albums/data/{album_id}/{artist_id}/primary.jpg'

      # Comment this back in to skip albums that don't have a primary image
      # if not os.path.exists(image_path):
      #   continue

      print(title)
      print(main_genre)
      print(image_path)
      print()