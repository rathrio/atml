import logging
import os
import pickle
from multiprocessing import freeze_support

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ISE_Pytorch.evaluation import get_embedding
from Agent.dataset_loaders import  MusicDataset, make_stratified_splits, PairwiseRankingLoss as latent_dist
from ISE_Pytorch.model import Img_Sen_Artist_Ranking
# changed configuration to this instead of argparse for easier interaction
CUDA = True
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 500

# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
ZDIMS = 40

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}

# load downloaded  dataset
# shuffle data at every epoch
# load dataset
music_dataset = MusicDataset(r"C:\Users\alvin\PycharmProjects\atml\data/metadata.csv", r'C:\Users\alvin\PycharmProjects\pytorch-skipthoughts/music_alb.npy' ) #if args are needed
#train_index, val_index = make_stratified_splits(music_dataset)
train_loader = DataLoader(music_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader =  DataLoader(music_dataset, batch_size=16, shuffle=True, **kwargs)

class CVAE(nn.Module):
    def __init__(self, save_loc):
        super(CVAE, self).__init__()

        # ENCODER
        # 1000 input vector from ISE, 512 outputs
        self.fc1 = nn.Linear(1000, 512)
        self.fc11 = nn.Linear(512,256)


        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(256, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(256, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections

        # DECODER( conditional decoder here)
        # from bottleneck to artist/genre dim
        self.fc3 = nn.Linear(ZDIMS, 196)
        self.fc4 = nn.Linear(196, 784)
        self.fc5 = nn.Linear(784, 1000)
        self.fc31 = nn.Linear(ZDIMS, 196)
        self.fc41 = nn.Linear(196, 700)
        self.fc51 = nn.Linear(700, 1000)
        self.sigmoid = nn.Sigmoid()

        #init ISE model tobe used later by agent
        if os.path.exists(save_loc ):
            logging.info('loading. options..' + save_loc)
            with open('%s/f30k_params_lstm.pkl' % save_loc, 'rb') as f:
                model_options = pickle.load(f)

        self.itag_model = Img_Sen_Artist_Ranking(model_options) # an instance of itag
        #load saved weights
        assert os.path.exists('%s/f30k_model_lstm.pkl' % save_loc)
        logging.info('Loading model...')

        self.itag_model.load_state_dict(torch.load('%s/f30k_model_lstm.pkl' % save_loc))
        logging.info('model loading done')
        self.itag_model.cuda()
        #turn traning off
        self.itag_model.eval()
        for param in self.itag_model.parameters():
            param.requires_grad = False

    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> reduce -> ReLU -> reduce, reduce)
        1000 ->512 ->256->40
        Returns
        -------
        (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS
            variance units one for each latent dimension
        """
        x= self.fc1(x)
        h1 = self.relu(self.fc11(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        For each training sample batch

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decodes to  the input distribution
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:

            return mu

    def decode_genre(self, z: Variable) -> Variable: #return some dim vector
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.relu(self.fc5(z))
        return self.sigmoid(z)

    def decode_artist(self, z: Variable) -> Variable: #return some dim vector
        z= self.fc31(z)
        z= self.fc41(z)
        z= self.relu(self.fc51(z))
        return self.sigmoid(z)

    def forward(self, x: Variable) -> (Variable, Variable, Variable,Variable): # we pass a vgg some vgg features

        # input size is 1000 dim of embbed output
        x_emb = self.itag_model.linear(x) # returns 1000 dim vector
        x_emb = F.normalize(x_emb)
        mu, logvar = self.encode(x_emb.view(-1, 1000))# should normalize
        z = self.reparameterize(mu, logvar)
        return self.decode_artist(z), self.decode_genre(z), mu, logvar


model = CVAE(r'C:\Users\alvin\PycharmProjects\atml\ISE_Pytorch\vse').cuda()

def loss_function(genres, artists, dec_artist, dec_genre, mu, logvar) -> Variable:

    #pass through pretrained sytem and get embedding
    #do forward pass of ISE here for embedding
    genre_emb=  get_embedding (model.itag_model, genres)
    artist_emb= get_embedding(model.itag_model, artists)

    lp = latent_dist()
    # how well do input data form itag and decoded vector from agent agree?
    BCE = lp(dec_artist, dec_genre,   artist_emb, genre_emb ) # make loss for decoders here

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * 1000

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

optimizer = optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4, mode='min',
                                      verbose=True, threshold=1e-8)

def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):

        vgg_feature, artist, genre = data
        vgg_feature= vgg_feature.cuda()


        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        #model take the vgg and make embedding itself and return a new generated vector.

        decoded_artist, decoded_genre,  mu, logvar = model(vgg_feature)
        # calculate scalar loss
        #get the embedded vectors of genre and artist for conditional decoding here
        loss = loss_function(genre,artist,decoded_artist, decoded_genre, mu, logvar)
        # calculate the distance with vae space, also our embedding

        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(vgg_feature), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item() / len(vgg_feature)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def validation(epoch):
    # toggle model to test / inference mode
    model.eval()
    test_loss = 0


    # each data is of BATCH_SIZE (default 128) samples
    for i, data in enumerate(test_loader):
        vgg_feature, artist, genre = data
        vgg_feature = vgg_feature.cuda()

        with torch.no_grad():

            decoded_artist, decoded_genre, mu, logvar = model(vgg_feature)
            test_loss += loss_function(genre,artist,decoded_artist, decoded_genre, mu, logvar)
            #n = min(data[0].size(0), 8) #get first 8 pics
            # for the first 128 batch of the epoch, show the first 8 input digits

            #todo for semantics reasoning
            #use the classifier to determine their respective artist, and genre
            #then also use it on the decoded vectors, then compare

    test_loss.data /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

tmp=0
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    val_loss= validation(epoch)
    if val_loss < tmp:
        torch.save(model.state_dict(), 'tesnsor.pt')
        tnp=val_loss
    scheduler.step(val_loss)


    '''#after training of agent call a random sample
    sample = Variable(torch.randn(64, ZDIMS)) #64 random items frm latent space

    sample = sample.cuda()
    sample = model.decode(sample)
    #use classifier to find close people'''

    #to visualize our latent space of 512 dim vector, lets use a trained classifier

if __name__ == '__main__':
        freeze_support()