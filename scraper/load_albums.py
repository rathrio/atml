#!/usr/bin/env python3

import os
import sys
import csv

# Place this script at /var/tmp/albums/load_albums.py

titles_file = open('/var/tmp/albums/titles.txt', 'w')

# To test locally
# titles_file = open('./titles.txt', 'w')

with open("data/metadata.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader, None) # skip headers

    for row in reader:
      album_id = row[0]
      title = row[1]
      artist_id = row[3]
      genres = row[5]
      main_genre = genres.split("|")[0]
      image_path = f'/var/tmp/albums/data/{album_id}/{artist_id}/primary.jpg'

      # Skip albums that don't have a primary image. Comment this out if you
      # want to test locally.
      if not os.path.exists(image_path):
        continue

      # Get features and dump feature
      # TODO OTI

      # Write title "sentence"
      titles_file.write(f'{title} .\n')