import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
from .dataset_loaders import  MusicDataset, make_stratified_splits, PairwiseRankingLoss as latent_dist
from ISE_Pytorch import  get_embedding
# changed configuration to this instead of argparse for easier interaction
CUDA = True
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 10

# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
ZDIMS = 20


torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# load downloaded  dataset
# shuffle data at every epoch
# load dataset
music_dataset = MusicDataset() #if args are needed
train_index, val_index = make_stratified_splits(music_dataset)
train_loader = DataLoader(music_dataset, batch_size=BATCH_SIZE,sampler=SubsetRandomSampler(train_index), **kwargs)
test_loader =  DataLoader(music_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_index), **kwargs)

class CVAE(nn.Module):
    def __init__(self):
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
        self.fc3 = nn.Linear(ZDIMS, 96)
        self.fc4 = nn.Linear(96, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc31 = nn.Linear(ZDIMS, 96)
        self.fc41 = nn.Linear(96, 256)
        self.fc51 = nn.Linear(256, 512)
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, x: Variable) -> (Variable, Variable, Variable,Variable):
        mu, logvar = self.encode(x.view(-1, 1000))
        z = self.reparameterize(mu, logvar)
        return self.decode_artist(z), self.decode_genre(z), mu, logvar


model = CVAE().cuda()


def loss_function(generated_artist, generated_genre, mu, logvar) -> Variable:
    #find the index of the vggfeatures sent to system, get artist and genre
    #pass through pretrained sytem and get embedding
    #do forward pass of ISE here for embedding
    #todo here
    vf,artists, genres = train_loader[i]
    assert vf == vggfeatures_sent

    genre_emb= get_embedding(genres)
    artist_emb= get_embedding(artists)


    # how well do input x and output recon_x agree?
    BCE = latent_dist(generated_artist, generated_genre,  genre_emb, artist_emb) # make loss for decoders here

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * 784

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        decoded_artist, decoded_genre,  mu, logvar = model(data)
        # calculate scalar loss
        loss = loss_function(decoded_artist, decoded_genre, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
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


def validation(epoch):
    # toggle model to test / inference mode
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE (default 128) samples
    for i, (data, _) in enumerate(test_loader):
        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()

        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          # for the first 128 batch of the epoch, show the first 8 input digits
          # with right below them the reconstructed output digits
          comparison = torch.cat([data[:n],
                                  recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, EPOCHS + 1):
    train(epoch)

    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space
    sample = Variable(torch.randn(64, ZDIMS))
    if CUDA:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()

    # save out as an 8x8 matrix of MNIST digits
    # this will give you a visual idea of how well latent space can generate things
    # that look like digits
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')