#!/usr/bin/env python3

import csv

# Place this script at /var/tmp/albums/generate_title_artist_genre.py

titles_file = open('/var/tmp/albums/music_sentences.txt', 'w')

with open("data/metadata.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader, None)  # skip headers

    for row in reader:
        title = row[1]
        artist = row[2]
        genre = row[5].split("|")[0]


        print(title)
        print(artist)
        print(genre)