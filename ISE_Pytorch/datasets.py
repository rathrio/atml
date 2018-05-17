"""
Dataset loading
"""
import numpy
import csv


def load_dataset(name='f8k'):
    """
    Load captions and image features, artist, genre
    """
    # loc =  name + '/'
    # # titles
    # titles,genre,artist = [],[],[]
    # with open(loc+name+'all_data.txt', 'rb') as f: #need loc to file here
    #     for line in f:
    #         #get the titles, genre and artist
    #         titles.append(line.strip())

    titles = []
    artists = []
    genres = []

    with open("/var/tmp/albums/data/metadata.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None)  # skip headers

        for row in reader:
            title = row[1]
            artist = row[2]
            genre = row[5].split("|")[0]
            titles.append(title)
            artists.append(artist)
            genres.append(genre)

            # Image features
            #train_ims = numpy.load(loc+name+'_train_ims.npy')
    album_ims = numpy.load('/var/tmp/albums/music_alb.npy')

    return (titles, album_ims, artists, genres)
