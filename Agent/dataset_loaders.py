import logging
import os
import pickle

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

from torchvision import transforms


from metadata import Metadata


test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
        self.metadata = Metadata(path_to_metadata)

        self.path_to_vggfeatures = path_to_vggfeatures

        self.albums = self.metadata.albums
        self.artist_name = ''

        self.vggfeatures = np.load(path_to_vggfeatures)
        print(f'Loaded {self.vggfeatures.shape[0]} vggfeatures')

    def __getitem__(self, index):
        #how to get a single item
        #What is needed is music vgg features extracted, artist name and genre

        album = self.albums[index]
        self.artist_name = album.artist
        genre = album.genre
        image_path = album.image_path

        image = Image.open(image_path).convert('RGB')

        image = test_transform(image)

        return self.vggfeatures[index], self.artist_name, genre, image

    def __len__(self):
        return self.vggfeatures.shape[0]  # how to get length of the dataset


def make_stratified_splits(dataset):
        x = dataset.vggfeatures
        y = dataset.artist_name  # startify is based on labels
        test_straf = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, train_size=0.8, random_state=4456)
        train_index, val_index = next(test_straf.split(x, y))

        # we can equiv also retrn these indexes for the random sampler to do its job
        # print(test_index,train_index,val_index)
        return train_index,  val_index

# dataset = MusicDataset('./data/metadata.csv', './data/music_alb.npy')
# dataset.__getitem__(12347)


class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, vgg, artist, genre, target_vgg, target_art, target_gen):
        margin = 0.2
        # compute vector-vector score matrix of generated and expected
        #between artist and genres from vgg
        #s = batch,l2norms_vab_size

        score_vgg = torch.mm(artist, target_art.transpose(1, 0))
        score_artist = torch.mm(artist, target_art.transpose(1, 0))
        score_genre = torch.mm(genre, genre.transpose(1, 0))

        diagonalv, diagonala, diagonalg = score_vgg, score_artist.diag(), score_genre.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive generated vectors for vectors)
        cost_vgg = torch.max(Variable(torch.zeros(score_vgg.size()[0], score_vgg.size()[1]).cuda()),
                             (margin - diagonala).expand_as(score_vgg) + score_vgg)

        cost_artist = torch.max(Variable(torch.zeros(score_artist.size()[0], score_artist.size()[1]).cuda()),
                                (margin-diagonala).expand_as(score_artist)+score_artist)

        cost_genre = torch.max(Variable(torch.zeros(score_genre.size()[0], score_genre.size()[1]).cuda()),
                               (margin - diagonalg).expand_as(score_genre) + score_genre)

        for i in range(score_artist.size()[0]):
            cost_vgg[i, i] = 0
            cost_artist[i, i] = 0
            cost_genre[i, i] = 0

        return cost_artist.sum() + cost_genre.sum() + cost_vgg.sum()
    #return the cost for the distances between the generated features , and the features of some layer of same size for
    #used for classification