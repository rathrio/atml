# Notice that in this assignment you are not allowed to use torch optimizer and nn module.
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from math import log10
import  matplotlib.pyplot as plt

# Hyper Parameters
from torch.autograd import Variable

input_size = 784
hidden_sizes = [ 64, 128, 192, 256, 320, 384, 448, 512,576, 640, 768,896,1024]
num_epochs = 10
batch_size = 100
learning_rate = 1e-5
psnr_list = []

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
#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
# add cuda option

for hidden_size in hidden_sizes:
    w1 = Variable(torch.randn(input_size, hidden_size).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(hidden_size, input_size).type(dtype), requires_grad=True)
    # Train the Model

    for epoch in range(num_epochs):
        for i, (images, _,) in enumerate(train_loader):

            images = Variable(images.view(-1, 28 * 28), requires_grad=False).cuda()
            targets = images.clone()

            # Forward : compute predicted y

            y_pred = images.mm(w1).clamp(min=0).mm(w2)

            # loss calculation
            loss = (y_pred - targets).pow(2).sum()

            # gradient calculation and update parameters
            # Backprop to compute gradients of w1 and w2 with respect to loss
            loss.backward()
            # Update weights using gradient descent
            w1.data -= learning_rate * w1.grad.data
            w2.data -= learning_rate * w2.grad.data

            # check your loss
            '''if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
                # Manually zero the gradients after running the backward pass'''
            w1.grad.data.zero_()
            w2.grad.data.zero_()
        print(epoch, '--', loss.data[0])

    # Test the Model
    # make the model , new function with waits and stuffs( a copy of forward pass with the w's)
    avg_psnr = 0
    w1 = w1.data.cpu()
    w2 = w2.data.cpu()
    for (images, _,) in test_loader:
        images = images.view(-1, 28 * 28)
        targets = images.clone()

        predictions = images.mm(w1).clamp(min=0).mm(w2)
        # calculate PSNR
        mse = torch.mean((predictions - targets).pow(2))
        psnr = 10 * log10(1 / mse)
        avg_psnr += psnr
    print(hidden_size, "===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))
    # save the sizes to loss ratio
    psnr_list.append(avg_psnr / len(test_loader))
    # loss list
    # size list

# make the graphs here.
plt.plot(hidden_sizes, psnr_list)
plt.title('Plot of Hidden size vs PSNR')
plt.ylabel('PSNR in db')
plt.xlabel('Hidden size')
plt.xticks(hidden_sizes,['64', '128', '192', '256', '320', '384', '448', '512','576', '640', '768','896','1024'] )
#plt.savefig(name+str(randint(0,1000))+'.png')
#plt.close()
plt.show()