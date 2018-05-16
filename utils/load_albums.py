#!/usr/bin/env python3

import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torchvision import  models, transforms
from torch.autograd import Variable
from PIL import Image

# Place this script at /var/tmp/albums/load_albums.py

titles_file = open('/var/tmp/albums/titles.txt', 'w')

# To test locally
# titles_file = open('./titles.txt', 'w')

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")


test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


vgg_model = models.vgg19_bn(pretrained=True)
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-3])
vgg_model.cuda()
vgg_model.eval()

i = 0
with open("data/metadata.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader, None) # skip headers
    feature_list = []

    for row in reader:
          if i == 3:
            break

          album_id = row[0]
          title = row[1]
          artist_id = row[3]
          genres = row[5]
          main_genre = genres.split("|")[0]
          image_path = f'/var/tmp/albums/data/{album_id}/{artist_id}/primary.jpg'

          # Skip albums that don't have a primary image. Comment this out if you
          # want to test locally.
          if not os.path.exists(image_path):
            continue

          # Write title "sentence"
          titles_file.write(f'{title} .\n')

          # Get features and dump feature
          file_path = ''

          if (is_image(file_path)):
              image = Image.open(file_path).convert('RGB')
              with torch.no_grad():  # put in no graph save mode
                  image = test_transform(image)
                  inputs = image.unsqueeze(0)
                  inputs = Variable(inputs).cuda()
                  # inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))  # add batch dim in the front
                  features = vgg_model(inputs)
                  feature_list.append(features)

          i += 1

          # save with numpy here
    np.save('/var/tmp/albums/music_alb.npy', feature_list)