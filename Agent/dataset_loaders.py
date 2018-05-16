from torch.utils.data import Dataset
import numpy as np
import csv

# Place this at /var/tmp/albums/dataset_loaders.py or adjust the passed in
# paths accordingly.

class MusicDataset(Dataset):
    """An override of class Dataset.

    All  datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, path_to_metadata, path_to_vggfeatures):
        #some arguments needed for music dataset to function?
        #place init of them here, like csv, folder_loc, etc
        self.path_to_metadata = path_to_metadata
        self.path_to_vggfeatures = path_to_vggfeatures

        self.albums = []

        with open("data/metadata.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip headers
            for row in reader:
                self.albums.append(row)

        print(f'Loaded {len(self.albums)} albums')

        self.vggfeatures = np.load(path_to_vggfeatures)
        print(f'Loaded {self.vggfeatures.shape[0]} vggfeatures')

    def __getitem__(self, index):
        #how to get a single item
        #What is needed is music vgg features extracted, artist name and genre

        row = self.albums[index]
        artist_name = row[2]
        genre = row[5].split("|")[0]

        return self.vggfeatures[index], artist_name, genre

    def __len__(self):
        return self.vggfeatures.shape[0] # how to get length of the dataset

# dataset = MusicDataset('./data/metadata.csv', './data/music_alb.npy')
# dataset.__getitem__(12347)



