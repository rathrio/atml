"""
Dataset loading
"""
import numpy



def load_dataset(name='f8k'):
    """
    Load captions and image features, artist, genre
    """
    loc =  name + '/'
    # titles
    titles,genre,artist = [],[],[]
    with open(loc+name+'all_data.txt', 'rb') as f: #need loc to file here
        for line in f:
            #get the titles, genre and artist
            titles.append(line.strip())



    # Image features
    #train_ims = numpy.load(loc+name+'_train_ims.npy')
    album_ims = numpy.load('music_alb.npy')

    return (titles, album_ims,artist, genre)