#here we get our model
#make some functin to accept some inputs and return some outputs vectors#
#compare how close they are to extracted features that can tell actual information.

from torchvision import  models, transforms
from PIL import Image
import torch.nn as nn
import torch
from torch.autograd import Variable
import pickle
from ISE_Pytorch.model import Img_Sen_Artist_Ranking
from ISE_Pytorch.evaluation import get_embedding

import numpy as np

class Evaluation:

    cuda_available = torch.cuda.is_available()

    def is_image(self, f):
        return f.endswith(".png") or f.endswith(".jpg")

    def getVggFeatures(self, file_path):

        vgg_model = models.vgg19_bn(pretrained=True)
        vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-3])
    
        if self.cuda_available:
            vgg_model = vgg_model.cuda()
    
        vgg_model.eval()

        if (self.is_image(file_path)):
            image = Image.open(file_path).convert('RGB')
            
            image = test_transform(image)
            inputs = image.unsqueeze(0)
            inputs = Variable(inputs)
            
            if self.cuda_available:
                inputs = inputs.cuda()

            features = vgg_model(inputs)

        return features

    def getImageTitleArtistGenreEmbeddings(self, vgg_features, title, artist, genre):
        
        print('Loading ITAG options...')
        with open('f30k_params_lstm.pkl', 'rb') as f:
            model_options = pickle.load(f)
        print('Options loaded.')
       
        print('Loading ITAG model...')
        itag_model = Img_Sen_Artist_Ranking(model_options)
        itag_model.load_state_dict(torch.load('f30k_model_lstm.pkl'))
        print('Model loaded.')

        if self.cuda_available:
            itag_model = itag_model.cuda()

        itag_model.eval()

        if self.cuda_available:
            vgg_features = vgg_features.cuda()

        img_emb = itag_model.linear(vgg_features)
        title_emb = get_embedding(itag_model, title)
        artist_emb = get_embedding(itag_model, artist)
        genre_emb =  get_embedding (itag_model, genre)

        return title_emb, img_emb, artist_emb, genre_emb

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

    #Album Info
    title = ""
    artist = ""
    genre = ""

    #Calculate features
    vgg_features = e.getVggFeatures(file_path)
    
    #Test with found features
    #t = np.load("music_alb.npy")
    #a = np.squeeze(vgg_features.data.numpy())
    #print(np.array_equal(t[5], a)) - True

    #Get ITAG Embeddings
    title_emb, img_emb, artist_emb, genre_emb = e.getImageTitleArtistGenreEmbeddings(vgg_features, title, artist, genre)


