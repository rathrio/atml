# Notice that in this assignment you are not allowed to use torch optimizer and nn module.
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from math import log10

# Hyper Parameters
from torch.autograd import Variable

input_size = 784
hidden_size = 128
num_epochs = 10
batch_size = 100
learning_rate = 1e-4

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# initialize your parameters: Randomly initialize weights
dtype = torch.FloatTensor
# add cuda option


w1 = torch.randn(input_size, hidden_size).type(dtype)
w2 = torch.randn(hidden_size, input_size).type(dtype)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, _,) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = images.view(-1, 28*28)
        targets = images.clone()

        # Forward : compute predicted y
        h1 = images.mm(w1)

        #implement the non function
        non = h1.clamp(min=0)
        #non =  h1[h1 < 0] = 0
        y_pred = non.mm(w2)


        # loss calculation
        loss = (y_pred - targets).pow(2).sum()

        # gradient calculation and update parameters
        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - targets) # top
        grad_w2 = non.t().mm(grad_y_pred)# last layer
        grad_h1_non = grad_y_pred.mm(w2.t()) #1
        grad_non = grad_h1_non.clone()
        grad_non[h1 < 0] = 0
        grad_w1 = images.t().mm(grad_non)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        
        # check your loss
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss))

# Test the Model
#make the model , new function with waits and stuffs( a copy of forward pass with the w's)
avg_psnr = 0
for (images, _,) in test_loader:
    images = images.view(-1, 28*28)
    targets = images.clone()

    predictions = images.mm(w1).clamp(min=0).mm(w2)
    # calculate PSNR
    mse = torch.mean((predictions - targets).pow(2))
    psnr = 10 * log10(1 / mse)
    avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))


#make the graphs here.
