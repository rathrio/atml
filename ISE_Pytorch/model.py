# coding: utf-8
import torch
from utils import l2norm, xavier_weight, init_lstm
from torch.autograd import Variable
import torch.nn.init as I


class Img_Sen_Artist_Ranking(torch.nn.Module):
    def __init__(self, model_options):
        super(Img_Sen_Artist_Ranking, self).__init__()
        self.linear = torch.nn.Linear(model_options['dim_image'], model_options['dim'])
        self.lstm = torch.nn.LSTM(model_options['dim_word'], model_options['dim'], 1)
        self.embedding = torch.nn.Embedding(model_options['n_words'], model_options['dim_word'])
        self.model_options = model_options
        self.init_weights()

    def init_weights(self):
        xavier_weight(self.linear.weight)
        self.linear.bias.data.fill_(0)
        #lstm weight init
        init_lstm(self.lstm)

    def forward(self, x, im,artist,genre):#titles, album_pic, artist, genre
        x_emb = self.embedding(x) #300 dim vector
        im = self.linear(im) # 1,000 dim vector
        artist_emb = self.embedding(artist)
        genre_emb = self.embedding(genre)

        _, (x_emb, _) = self.lstm(x_emb) # 1,000 dim vector
        _, (artist_emb, _) = self.lstm(artist_emb)
        _, (genre_emb, _) = self.lstm(genre_emb)

        x_emb, artist_emb, genre_emb = x_emb.squeeze(0), artist_emb.squeeze(0),genre_emb.squeeze(0)

        return l2norm(x_emb), l2norm(im), l2norm(artist_emb), l2norm(genre_emb) # batch, distance

    def forward_sens(self, x):
        x_emb = self.embedding(x)

        _, (x_emb, _) = self.lstm(x_emb)
        x_cat = x_emb.squeeze(0)
        return l2norm(x_cat)

    def forward_imgs(self, im):
        im = self.linear(im)
        return l2norm(im)




class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s, artist,genre):#
        margin = self.margin
        # compute image-sentence score matrix
        #s = batch,l2norms_vab_size

        scores = torch.mm(im, s.transpose(1, 0))
        scoreaa = torch.mm(im, artist.transpose(1, 0))
        scoregg = torch.mm(im, genre.transpose(1, 0))

        diagonal,diagonal1,diagonal2,  = scores.diag(), scoreaa.diag(), scoregg.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        #cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        #  all contrastive sentences for each image(take some margin of the l2norm of the sentence of image, and add to non-sentence scores)
        #non-setence will have high values which we wish to set to zero making score of image sentence high
        cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)
        #  all contrastive artist for each image
        cost_aa = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
                            (margin - diagonal1).expand_as(scores).transpose(1, 0) + scores)
        # all contrastive genre for each image
        cost_gg = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
                            (margin - diagonal2).expand_as(scores).transpose(1, 0) + scores)

        for i in range(scores.size()[0]):
            #cost_s[i, i] = 0
            cost_im[i, i] = 0
            cost_aa[i, i] = 0
            cost_gg[i, i] = 0

        return cost_aa.sum() + cost_im.sum() +cost_gg.sum() #+cost_s.sum()
