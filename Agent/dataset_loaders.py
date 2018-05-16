from torch.utils.data import Dataset


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



    def __getitem__(self, index):
        #how to get a single item
        #What is needed is music vgg features extracted, artist name and genre

        return #vggfeatures, artist_name, genre

    def __len__(self):
        return # how to get length of the dataset

dataset = MusicDataset('../data/metadata.csv', 'foobar')



