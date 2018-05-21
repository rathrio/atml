#here we get our model
#make some functin to accept some inputs and return some outputs vectors#
#compare how close they are to extracted features that can tell actual information.
import numpy
from torchvision import  models, transforms
from PIL import Image
import torch.nn as nn
import torch
from torch.autograd import Variable
import pickle

from ISE_Pytorch.datasets import load_dataset
from ISE_Pytorch.model import Img_Sen_Artist_Ranking
from ISE_Pytorch.tools_ import embbed_sentences
from ISE_Pytorch.evaluation import get_embedding

import numpy as np

class Evaluation:

    cuda_available = torch.cuda.is_available()

    #initualize the itag model here
    def __init__(self):
        #self.itag = Img_Sen_Artist_Ranking()
        print('Loading ITAG options...')
        with open(r'C:\Users\alvin\PycharmProjects\atml\ISE_Pytorch\vse\f30k_params_lstm.pkl', 'rb') as f:
            model_options = pickle.load(f)
        print('Options loaded.')

        print('Loading ITAG model...')
        self.itag_model = Img_Sen_Artist_Ranking(model_options)
        self.itag_model.load_state_dict(torch.load(r'C:\Users\alvin\PycharmProjects\atml\ISE_Pytorch\vse\f30k_model_lstm.pkl'))
        print('Model loaded.')

        if self.cuda_available:
            itag_model = self.itag_model.cuda()

        self.itag_model.eval()

    def is_image(self, f):
        return f.endswith(".png") or f.endswith(".jpg")

    def getVggFeatures(self, file_path):

        vgg_model = models.vgg19_bn(pretrained=True)
        vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-3])
    
        if self.cuda_available:
            vgg_model = vgg_model.cuda()
    
        vgg_model.eval()
        image = Image.open(file_path).convert('RGB')

        image = test_transform(image)
        inputs = image.unsqueeze(0)
        inputs = Variable(inputs)

        if self.cuda_available:
            inputs = inputs.cuda()

        features = vgg_model(inputs)

        return features

    def input_2_title(self, file_path ):
        vgg_features = e.getVggFeatures(file_path)
        #get the new data, and embbed it singlely

        #embedd all the captions with pretrained models
        #to pass to i2t
        #Pass the new data through the pipeline
        print('loading dataset')

        #all titles genre and artist
        titles, _,artist,genre = load_dataset('dummy')

        print('Done')

        #if u have a model, u can call embbed esentences to return the embedding
        titles_emb = embbed_sentences(self.itag_model, titles)
        artist_emb = embbed_sentences(self.itag_model, artist)
        genre_emb = embbed_sentences(self.itag_model, genre)
        print('dataset embedded')

        #Now we have an embedding of all titles,artist,genre to do compare with output
        #from our input embedded
        if self.cuda_available:
            vgg_features = vgg_features.cuda()
        print('embbed input image')
        img_emb = self.itag_model.linear(vgg_features)
        #embedding of the new input in the learned feature space

        # Compute scores
        title_score = torch.mm(img_emb, titles_emb.t())
        genre_score = torch.mm(img_emb, genre_emb.t())
        artist_score = torch.mm(img_emb, artist_emb.t())# multiply embedding vector of image with vectors of titles
        _,titles_inds = torch.sort(title_score, descending=True)
        _, genre_inds = torch.sort(genre_score, descending=True)
        _, artist_inds = torch.sort(artist_score, descending=True)
        #useget highest vector and its index is the prediction
        titles_inds = titles_inds.data.squeeze(0).cpu().numpy()[0:5]
        genre_inds = genre_inds.data.squeeze(0).cpu().numpy()[0:5]
        artist_inds = artist_inds.data.squeeze(0).cpu().numpy()[0:5]
        #take top 5 ranks
        for  i,(title,artist,genre) in enumerate(zip(titles_inds, artist_inds,genre_inds)):
            print('R@5=titles\t','artist\t','genre\t')
            print(titles[titles_inds[i]]+'\t', artist[artist_inds[i]]+'\t', genre[genre_inds[i]]+'\t')

        #todo debug artist and tille names
        return



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
    file_path = r"C:\Users\alvin\Pictures\metal.jpg"

    #Album Info
    artist = ""
    genre = ""

    #Calculate features

    title= e.input_2_title(file_path)
    #print()
    #call itag here to get some tiles and stuff
    #use skipthought to get another sentece


