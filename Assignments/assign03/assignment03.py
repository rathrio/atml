import time
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import sys


# Set Hyperparameters
noise_type = None  # Possible values: 'gaussian_add', 'noise_salt_pepper', 'noise_masking' or None
finetune = True
num_epochs = 10
batch_size = 128
learning_rate = 0.001
LAYER_DIMS = [16, 8, 8]

# Check if we can use CUDA
cuda_available = torch.cuda.is_available()

# Define image transformations & Initialize datasets
mnist_transforms = transforms.Compose([transforms.ToTensor()])
mnist_train = dset.MNIST('./data', train=True, transform=mnist_transforms, download=True)
mnist_test = dset.MNIST('./data', train=False, transform=mnist_transforms, download=True)

# For reproducibility
torch.manual_seed(123)
np.random.seed(123)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True)
testloader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)

# Choose 5000 examples for transer learning
mask = np.random.randint(0, 60000, 5000)
finetune_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=SubsetRandomSampler(np.where(mask)[0]),
                                              num_workers=2)


# Create Encoder and Decoder that subclasses nn.Module
class Encoder(nn.Module):
    """Convnet Encoder"""

    def __init__(self):
        super(Encoder, self).__init__()
        # 28 x 28 -> 14 x 14
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=LAYER_DIMS[0], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=LAYER_DIMS[0]),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # 14 x 14 -> 7 x 7
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=LAYER_DIMS[0], out_channels=LAYER_DIMS[1], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=LAYER_DIMS[1]),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # 7 x 7 -> 4 x 4
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=LAYER_DIMS[1], out_channels=LAYER_DIMS[2], kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=LAYER_DIMS[2]),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class Decoder(nn.Module):
    """Convnet Decoder"""

    def __init__(self):
        super(Decoder, self).__init__()
        # 4 x 4 -> 7 x 7
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LAYER_DIMS[2], out_channels=LAYER_DIMS[1],
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(LAYER_DIMS[1]),
        )
        # 7 x 7 -> 14 x 14
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LAYER_DIMS[1], out_channels=LAYER_DIMS[0],
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(LAYER_DIMS[0]),
        )
        # 14 x 14 -> 28 x 28
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LAYER_DIMS[0], out_channels=1,
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


# Create a Classifer for the Encoder features
class Classifier(nn.Module):
    """Convnet Classifier"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=LAYER_DIMS[2], out_channels=10, kernel_size=(4, 4), padding=0)
        )
        self.lin = nn.Sequential(
            nn.Linear(10, 10)
        )

    def forward(self, x):
        out = self.classifier(x).squeeze()
        out = self.lin(out)
        return out


def noise_additive_gaussian(imgs, sigma=.5):
    """
    Adds additive gaussian noise to images for the training of a DAE

    Args:
        imgs: A batch of images
        sigma: Standard deviation of the gaussian noise

    Returns:
        imgs_n: The noisy images

    """
    imgs_n = None
    #######################################################################
    # TODO:                                                               #
    # Apply additive Gaussian noise to the images                         #
    #                                                                     #
    #######################################################################
    imgs_n = torch.normal(means=imgs, std=sigma)
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return imgs_n


def noise_salt_pepper(imgs, noise_rate=0.5):
    """
    Adds salt&pepper noise to images for the training of a DAE

    Args:
        imgs: A batch of images
        noise_rate: Controls the amount of noise (higher=more noise)

    Returns:
        imgs_n: The noisy images

    """
    imgs_n = None
    #######################################################################
    # TODO:                                                               #
    # Apply Salt&Pepper noise to the images                               #
    #                                                                     #
    #######################################################################
    imgs_n = imgs.clone()
    for row in imgs_n:
        dt = row.data.numpy()
        rsDt = dt.reshape(1,28*28) 
        idx = np.arange(rsDt.shape[1])
        np.random.shuffle(idx)
        chg = int(len(idx)*noise_rate)
        idx = idx[:chg]
        rsDt[:, idx[:int(chg/2)]]=0
        rsDt[:, idx[int(chg/2):]]=1
        row.data = torch.from_numpy(rsDt)
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return imgs_n


def noise_masking(imgs, drop_rate=0.5, tile_size=7):
    """
    Randomly sets tiles of images to zero for the training of a DAE

    Args:
        imgs: A batch of images
        drop_rate: Controls the amount of tile dropping (higher=more noise)
        tile_size: The size of the tiles to be dropped in pixels

    Returns:
        imgs_n: The noisy images

    """
    imgs_n = None
    #######################################################################
    # TODO:                                                               #
    # Apply masking to the images                                         #
    #                                                                     #
    #######################################################################
    imgs_n = imgs.clone()
    for row in imgs_n:
        dt = row.data.numpy()
        rsDt = dt.reshape(1,28*28) 

        idx = np.arange(rsDt.shape[1])
        np.random.shuffle(idx)
        chg = int(len(idx)*drop_rate/tile_size)
        idx = idx[:chg]
        idn = np.copy(idx)
        #arrange tiles
        for i in range(tile_size-1):
            idn = np.append(idn,idx+i+1)
        idn[idn>783] = 0        
        rsDt[:, idn]=0
        row.data = torch.from_numpy(rsDt)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return imgs_n


encoder = Encoder()
decoder = Decoder()
if cuda_available:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# Define Loss and Optimizer for DAE training
parameters = list(encoder.parameters()) + list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# Get noise function to be applied to images
if noise_type is 'gaussian_add':
    image_fn = noise_additive_gaussian
elif noise_type is 'noise_salt_pepper':
    image_fn = noise_salt_pepper
elif noise_type is 'noise_masking':
    image_fn = noise_masking
else:
    # Default is no noise (standard AE)
    image_fn = lambda x: x

print('--------------------------------------------------------------')
print('---------------------- Training DAE --------------------------')
print('--------------------------------------------------------------')

# Train the Autoencoder
for epoch in range(num_epochs):
    losses = []
    start = time.time()
    for batch_index, (image, _) in enumerate(train_loader):
        if cuda_available:
            image = image.cuda()
        image = Variable(image)
        image_n = image_fn(image)

        # Training Step
        optimizer.zero_grad()
        output = encoder(image_n)
        output = decoder(output)
        loss = loss_func(output, image)
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
        if batch_index % 50 == 0:
            print('Epoch: {}, Iter: {:3d}, Loss: {:.4f}'.format(epoch, batch_index, loss.data[0]))

    end = time.time()
    print('Epoch: {}, Average Loss: {:.4f}, Time: {:.4f}'.format(epoch, np.mean(losses), end - start))

# Set encoder and decoder in evaluation mode to use running means and averages for Batchnorm
encoder.eval()
decoder.eval()

# Get a batch of test images
test_imgs, test_labels = next(iter(testloader))
if cuda_available:
    test_imgs, test_labels = test_imgs.cuda(), test_labels.cuda()
test_imgs, test_labels = Variable(test_imgs), Variable(test_labels)
image_in = image_fn(test_imgs)

output = encoder(image_in)
output = decoder(output)

# Visualize in and output of the Autoencoder
fig_out = plt.figure('out', figsize=(10, 10))
fig_in = plt.figure('in', figsize=(10, 10))
for ind, (img_out, img_in) in enumerate(zip(output, image_in)):
    if ind > 63:
        break
    plt.figure('out')
    fig_out.add_subplot(8, 8, ind + 1)
    plt.imshow(img_out.data.cpu().numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.figure('in')
    fig_in.add_subplot(8, 8, ind + 1)
    plt.imshow(img_in.data.cpu().numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

print('--------------------------------------------------------------')
print('------------------- Transfer Learning ------------------------')
print('--------------------------------------------------------------')

#######################################################################
# TODO:                                                               #
# Prepare everything for transfer learning:                           #
#   - Build the classifier                                            #
#   - Define the optimizer                                            #
#   - Define the loss function                                        #
# Note: The setup might be diffent for finetuning or fixed features   #
#       (see variable finetune!)                                      #
#                                                                     #
#######################################################################
clf = Classifier()
#if finetune is true, then send encoder params for optimization else prevent encoder to optimize (use as fixed encoder)
params = list(clf.parameters()) + list(encoder.parameters()) if finetune else clf.parameters()
optimizer = torch.optim.Adam(params, lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
#######################################################################
#                         END OF YOUR CODE                            #
#######################################################################

# Train the Classifier
for epoch in range(30):
    losses = []
    start = time.time()
    for batch_index, (image, label) in enumerate(finetune_loader):
        if cuda_available:
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)

        # Training Step
        optimizer.zero_grad()
        output = encoder(image)
        output = clf(output)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])

    end = time.time()
    print('Epoch: {}, Average Loss: {:.4f}, Time: {:.4f}'.format(epoch, np.mean(losses), end - start))

    #######################################################################
    # TODO:                                                               #
    # Evaluate the classifier on the test set by computing the accuracy   #
    # of the classifier                                                   #
    #                                                                     #
    #######################################################################
    correct = 0
    for batch_index, (image, label) in enumerate(testloader):
        if cuda_available:
            image = image.cuda()
        image = Variable(image)
        
        output = encoder(image)
        output = clf(output)

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum()

    accuracy = correct / len(testloader.dataset)
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    print('Epoch: {}, Test Acc: {:.4f}'.format(epoch, accuracy))
    print('--------------------------------------------------------------')
    clf.train()
    if finetune:
        encoder.train()
