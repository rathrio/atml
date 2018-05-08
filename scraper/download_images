#!/usr/bin/env python3

import os
import sys
import csv
import urllib.request

with open("data/metadata.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader, None) # skip headers

    for row in reader:
        album_id = row[0]
        artist_id = row[3]
        primary_image = row[8]
        secondary_image = row[9]

        dir = f'data/{album_id}/{artist_id}'
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        location = f'{dir}/primary.jpg'

        if os.path.isfile(location):
            sys.stdout.write('✓')
            sys.stdout.flush()
            continue

        try:
            urllib.request.urlretrieve(primary_image, location)
        except:
            pass

        try:
            if primary_image:
                location = f'{dir}/secondary.jpg'
            urllib.request.urlretrieve(secondary_image, location)
        except:
            pass

        sys.stdout.write('.')
        sys.stdout.flush()

print("\nDONE")