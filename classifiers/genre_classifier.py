import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from classifiers.metadata import Metadata
from Agent.dataset_loaders import MusicDataset
from tensorboardX import SummaryWriter

import IPython

metadata_path = "./data/metadata.csv"
features_path = "./data/music_alb.npy"
save_model_path = "./data/models"


import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from Agent.dataset_loaders import MusicDataset_transformed
from Agent.AutoencoderNet import VAE as Pic_2vgg_2pic

CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 10


torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}

music_dataset = MusicDataset_transformed(r"C:\Users\alvin\PycharmProjects\atml\data/metadata.csv",
                             r'C:\Users\alvin\PycharmProjects\pytorch-skipthoughts/music_alb.npy' ) #if args are needed
#train_index, val_index = make_stratified_splits(music_dataset)
#make splits

num_train = len(music_dataset)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))

np.random.seed(12354)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(music_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, **kwargs)
test_loader =  DataLoader(music_dataset, batch_size=128, sampler=valid_sampler, **kwargs)


class VAE_genre(nn.Module):
    def __init__(self):
        super(VAE_genre, self).__init__()
        #load a pretrainned autoencoder on pic -> to vgg-> to pic
        #use the weights and put on a classifier
        self.pic_2vgg_2pic_encoder = Pic_2vgg_2pic()
        #load its state_dicts
        self.pic_2vgg_2pic_encoder.load_state_dict(torch.load('loc_to_saved_model'))

        print('Pic_2vgg_2pic_encoder model loaded...')

        self.pic_2vgg_2pic_encoder.load_state_dict(torch.load('%s/f30k_model_lstm.pkl' % save_model_path))
        print('model loading done')
        self.pic_2vgg_2pic_encoder.cuda()
        # turn traning off
        self.pic_2vgg_2pic_encoder.eval()
        for param in self.pic_2vgg_2pic_encoder.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(40,15) #todo add re-shaping logic




    def forward(self, x: Variable) -> (Variable, Variable):
        relu_out, out = self.pic_2vgg_2pic_encoder(x)
        #z = self.reparameterize(mu, logvar)
        out= self.fc1(out)
        relu_out = self.fc1(relu_out) # 15 dim vector
        return out, relu_out


model = VAE_genre()
if CUDA:
    model.cuda()

params = filter(lambda p: p.requires_grad, model.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr=1e-3, weight_decay=3e-3)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min',
                                      verbose=True, threshold=1e-8)


def train(epoch):

    model.train()
    train_out_loss = 0
    train_relu_out_loss = 0

    for batch_idx, data in enumerate(train_loader): #trainloader from Musicdataset return title,im,art,gen
        #rturn genre is transformed

        title, vgg_feature, artist, genre = data
        vgg_feature= vgg_feature.cuda()

        optimizer.zero_grad()

        out, relu_out = model(vgg_feature)
        loss_out = criterion(out, genre)
        loss_relu_out = criterion(relu_out, genre)
        loss_out.backward()
        loss_relu_out.backward()
        train_out_loss += loss_out.data.item()
        train_relu_out_loss += loss_relu_out.data.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\toutLoss: {:.6f}\treluedLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_out.data.item() / len(data),
                loss_relu_out.data.item() / len(data)))

    print('====> Epoch: {} Average losses: {:.4f}, {:.4f}'.format(
          epoch, train_out_loss / len(train_loader.dataset),
        train_out_loss / len(train_loader.dataset)))
    return train_out_loss, train_out_loss


def test(epoch):
    model.eval()
    test_out_loss, test_relu_out_loss = 0
    for batch_idx, data in enumerate(train_loader):

        _, vgg_feature, _, genre = data
        vgg_feature= vgg_feature.cuda()

        with torch.no_grad():

            out, relu_out = model(vgg_feature)
            test_out_loss += criterion(out, genre).data.item()
            test_relu_out_loss += criterion(relu_out, genre).data.item()


    test_out_loss /= len(test_loader.dataset)
    test_relu_out_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}, {:.4f}'.format(test_out_loss,test_relu_out_loss))
    return test_out_loss, test_relu_out_loss

writer = SummaryWriter()
tmp = 1e20
for epoch in range(1, EPOCHS + 1):
    train_out_loss, train_relu_out_loss = train(epoch)
    test_out_loss, test_relu_out_loss= test(epoch)

    # data grouping by `slash`
    #writer.add_scalar('Evaluation/trainingLoss', train_out_loss, epoch)
    #writer.add_scalar('Evaluation/Validation_set', val_loss, epoch)

    writer.add_scalars('Genre_Classifier_Evaluation/Train_Loss', {'Train_out': test_out_loss,
                                                         'Train_relu_out': test_relu_out_loss,
                                                         }, epoch)

    writer.add_scalars('Genre_Classifier_Evaluation/Test_Loss', {'Test_out': test_out_loss,
                                                         'Test_relu_out': test_relu_out_loss,
                                                         }, epoch)
    total_loss = test_out_loss+test_relu_out_loss
    if test_out_loss < tmp or test_relu_out_loss < tmp:
        best = max(test_out_loss, test_relu_out_loss)
        print('saving model @', best)
        torch.save(model.state_dict(), ('gemre_classifier_model@_%s.pt' % best))
        tmp=best
    scheduler.step(total_loss)








'''
cuda_available = torch.cuda.is_available()

class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, genre_count):
        super(GenreClassifier, self).__init__()
        self.genre_count = genre_count
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, genre_count, bias=False)

    def forward(self, features):
        out = self.fc1(features)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


def train():
    metadata = Metadata(metadata_path)
    albums = metadata.albums
    all_features = np.load(features_path)
    print(f'Loaded {len(all_features)} features')

    assert len(albums) == len(all_features)

    input_size = 4096
    hidden_size = 1000
    genre_count = metadata.genre_count
    net = GenreClassifier(input_size, hidden_size, genre_count)
    if cuda_available:
        net = net.cuda()

    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    batch_size = 1000

    # try:
    for epoch in range(10):

        running_loss = 0.0
        for i, album in enumerate(albums, 0):
            features = torch.Tensor([all_features[i].tolist()])
            if cuda_available:
                features = features.cuda()
            features = Variable(features)

            genre = torch.LongTensor([metadata.genre_index(album.genre)])
            if cuda_available:
                genre = genre.cuda()
            genre = Variable(genre)

            optimizer.zero_grad()

            out = net(features)
            loss = criterion(out, genre)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            running_loss += loss.data[0]
            if i % batch_size == (batch_size - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / batch_size))
                running_loss = 0.0
    # except:
    #     torch.save(net.state_dict(), f'{save_model_path}/genre_model')

    torch.save(net.state_dict(), f'{save_model_path}/genre_classifier.pkl')
    print('Done!')


# metadata = Metadata(metadata_path)
# albums = metadata.albums
# all_features = np.load(features_path)
# print(f'Loaded {len(all_features)} features')

# assert len(albums) == len(all_features)

# input_size = 4096
# hidden_size = 1000
# genre_count = metadata.genre_count
# net = GenreClassifier(input_size, hidden_size, genre_count)
# net.load_state_dict(torch.load('./classifiers/genre_classifier.pkl'))

# IPython.embed()
train()
'''

