import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from math import log10

# Hyper Parameters 
input_size = 784
hidden_size1 = 384
hidden_size2 = 512
num_epochs = 10
batch_size = 100
learning_rate = 0.001

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

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.fc3 = nn.Linear(hidden_size2, input_size)  

        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc3(x)

        return out
    
net = Net(input_size, hidden_size1, hidden_size2)

# print net
print(repr(net))
# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# Train the Model
for epoch in range(num_epochs):
    for i, (images, _, ) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        # # define your labels
        labels = images
        # forward + backward
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(images, labels)
        loss.backward()
        optimizer.step()

        # if (images - labels) == 0:

        #     predictions = t
        #     $

        # loss =
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    # Test the Model
    avg_psnr = 0
    for (images, _, ) in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = net(images)
        mse = criterion(outputs, images)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        # get predictions and calculate PSNR
