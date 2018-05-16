#!/usr/bin/env python3

import os
import sys
import csv
import random

# Place this script at /var/tmp/albums/generate_titles.py

titles_file = open('/var/tmp/albums/title_context.txt', 'w')
# To test locally
# titles_file = open('./title_context.txt', 'w')

with open("data/metadata.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader, None)  # skip headers

    for row in reader:
      title = row[1]
      songs = row[4].split("|")

      random.shuffle(songs)
      before = " ".join(songs)
      random.shuffle(songs)
      after = " ".join(songs)

      sentence = f'{before} {title} {after} .\n'
      titles_file.write(sentence)
