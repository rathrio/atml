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

    with open(r"C:\Users\alvin\PycharmProjects\atml\data/metadata.csv", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None)  # skip headers

        for i,row  in enumerate(reader):

            title = row[1]
            artist = row[2]
            genre = row[5].split("|")[0]
            titles.append(title)
            artists.append(artist)
            genres.append(genre)

            # Image features
            #train_ims = numpy.load(loc+name+'_train_ims.npy')
    album_ims = numpy.load(r'C:\Users\alvin\PycharmProjects\pytorch-skipthoughts/music_alb.npy')

    return (titles, album_ims, artists, genres)
