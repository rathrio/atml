
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

music_dataset = MusicDataset(r"C:\Users\alvin\PycharmProjects\atml\data/metadata.csv",
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



    def forward(self, x: Variable) -> (Variable, Variable):
        relu_out, out = self.encode(x)
        #z = self.reparameterize(mu, logvar)
        out= self.decode(out)
        relu_out = self.decode(relu_out)
        x_off = int((out.shape[2] - 224) / 2)
        y_off = int((out.shape[2] - 224) / 2)
        out = out[:, :, x_off:x_off + 224, y_off:y_off + 224]
        relu_out = relu_out[:, :, x_off:x_off + 224, y_off:y_off + 224]
        return out, relu_out


model = VAE()
if CUDA:
    model.cuda()


def loss_function(recon_x, x, mu, logvar) -> Variable:

    BCE = F.binary_cross_entropy(recon_x, x)

    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD /= BATCH_SIZE * 784
    return BCE #+ KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-3)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min',
                                      verbose=True, threshold=1e-8)


def train(epoch):

    model.train()
    train_out_loss = 0
    train_relu_out_loss = 0

    for batch_idx, data in enumerate(train_loader):

        vgg_feature, artist, genre, image_path = data #todo edit dataset loader file to give picture for comparison
        # To train the autoencoder to be able to generate sampled data
        vgg_feature= vgg_feature.cuda()
        #todo Album piture = here

        optimizer.zero_grad()

        out, relu_out = model(actual picture here)
        loss_out = loss_function(out, data)
        loss_relu_out = loss_function(relu_out, data)
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

        vgg_feature, artist, genre, image_path = data
        vgg_feature= vgg_feature.cuda()

        with torch.no_grad():

            out, relu_out = model(vgg_feature)
            test_out_loss += loss_function(out, actual_data).data.item()
            test_relu_out_loss += loss_function(relu_out, actual_data).data.item()
            '''if batch_idx == 0:
              n = min(data.size(0), 8)
              comparison = torch.cat([data[:n],
                                      out.view(BATCH_SIZE, 3, 224, 224)[:n]])
              save_image(comparison.data.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)'''

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

    writer.add_scalars('Evaluation/Train_Loss', {'Train_out': test_out_loss,
                                                         'Train_relu_out': test_relu_out_loss,
                                                         }, epoch)

    writer.add_scalars('Evaluation/Test_Loss', {'Test_out': test_out_loss,
                                                         'Test_relu_out': test_relu_out_loss,
                                                         }, epoch)
    total_loss = test_out_loss+test_relu_out_loss
    if test_out_loss < tmp or test_relu_out_loss < tmp:
        best = max(test_out_loss, test_relu_out_loss)
        print('saving model @', best)
        torch.save(model.state_dict(), ('auto_encoder_model@_%s.pt' % best))
        tmp=best
    scheduler.step(total_loss)

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

