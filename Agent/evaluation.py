#here we get our model
#make some functin to accept some inputs and return some outputs vectors#
#compare how close they are to extracted features that can tell actual information.

from torchvision import  models, transforms
from PIL import Image
import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy as np

class Evaluation:

    cuda_available = torch.cuda.is_available()

    vgg_model = models.vgg19_bn(pretrained=True)
    vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-3])
    
    if cuda_available:
        vgg_model.cuda()
    
    vgg_model.eval()

    def is_image(self, f):
        return f.endswith(".png") or f.endswith(".jpg")

    def getVggFeatures(self, file_path):
        if (self.is_image(file_path)):
            image = Image.open(file_path).convert('RGB')
            
            image = test_transform(image)
            inputs = image.unsqueeze(0)
            inputs = Variable(inputs)
            
            if self.cuda_available:
                inputs.cuda()

            features = self.vgg_model(inputs)

        return features

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Main Method
if __name__ == '__main__':
    
    # Evaluate Model
    e = Evaluation()

    #Image File Path
    file_path = "primary.jpg"

    #Calculate features
    vgg_features = e.getVggFeatures(file_path)
    
    #Test with found features
    #t = np.load("music_alb.npy")
    #a = np.squeeze(vgg_features.data.numpy())
    #print(np.array_equal(t[5], a)) - True




