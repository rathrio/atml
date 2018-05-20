import os
from multiprocessing import freeze_support
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# changed configuration to this instead of argparse for easier interaction
from Agent.dataset_loaders import MusicDataset

CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 10


torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}

music_dataset = MusicDataset(r"C:\Users\alvin\PycharmProjects\atml\data/metadata.csv", r'C:\Users\alvin\PycharmProjects\pytorch-skipthoughts/music_alb.npy' ) #if args are needed
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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()


        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 37632)

        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()


        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),  # 24x24x8
            nn.Conv2d(16, 28, 3, 1),  # 22x22x12
            nn.ReLU(inplace=True),  # 12x12x12
            nn.Conv2d(28, 32, 3, 1),  # 10x10x16
            nn.ReLU(inplace=True),  # 5x5x16
            nn.Conv2d(32, 40, 3, 2),  # 6x6x32

        )

        self.decoder = nn.Sequential( #todo 224*224*3 ->
            nn.ConvTranspose2d(40, 32, 5, stride=1),
            nn.ReLU(inplace=True),# 3x3x16
            nn.ConvTranspose2d(32, 28, 3, 2, padding=1, output_padding=1),  # 7x7x8
            nn.ConvTranspose2d(28, 16, 3, 1),  # 15x15x3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, 2),  # 32x32x1 for mnist
            nn.Sigmoid()
        )

    def encode(self, x : Variable) ->(Variable, Variable):

        x= self.fc1(x.squeeze(1).view(-1,4096)) #1000 dim to 784
        x = self.fc2(x)
        x= x.unsqueeze(1).view(-1,3, 112,112)# shape well for conv2d

        x = self.encoder(x)
        return x, self.relu(x)


    def decode(self, z: Variable) -> Variable:

        return self.decoder(z) #32x32x3

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:


        if self.training:

            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)

        else:

            return mu



    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x)
        #z = self.reparameterize(mu, logvar)
        out= self.decode(mu)
        x_off = int((out.shape[2] - 224) / 2)
        y_off = int((out.shape[2] - 224) / 2)
        out = out[:, :, x_off:x_off + 224, y_off:y_off + 224]
        return out, mu, logvar


model = VAE()
if CUDA:
    model.cuda()


def loss_function(recon_x, x, mu, logvar) -> Variable:

    BCE = F.binary_cross_entropy(recon_x, x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 784
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):

    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):

        vgg_feature, artist, genre = data #todo edit dataset loader file to give picture for comparison
        # To train the autoencoder to be able to generate sampled data
        vgg_feature= vgg_feature.cuda()
        #todo Album piture = here

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(actual picture here)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(train_loader):

        vgg_feature, artist, genre = data
        vgg_feature= vgg_feature.cuda()

        with torch.no_grad():

            data = Variable(data)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data.item()
            if batch_idx == 0:
              n = min(data.size(0), 8)
              comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
              save_image(comparison.data.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)

    ''' def forward(self, x):
        #x= nn.Linear(1000, 784)
        x= x.view(-1,788)
        en = self.encoder(x)
        out  = self.decoder(en)

        #get separate channels  from the decoder and lets hope they correspong to the signals
        #3 chanels, for album signal, genre signal, artist signal .

        vgg_signal, artist_signal, genre_signal = out[0],out[1],out[2]

        return vgg_signal, artist_signal,genre_signal


    def get_features(self, x):
        en = self.encoder(x)
        en  = en.view(-1, 32)
        return en'''

