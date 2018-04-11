

import torch
import torch.nn as nn

class AutoencoderNet(nn.Module):

    def __init__(self):
        super(AutoencoderNet, self).__init__()

        self.channel_num = 8
        self.encoder = nn.Sequential( #TODO : encoder architecture
            nn.Conv2d(in_channels= 1, out_channels=8, kernel_size= 5, stride=1), #24x24x8
            nn.Conv2d(8,12,3,1),# 22x22x12
            nn.MaxPool2d(2,2),# 12x12x12
            nn.Conv2d(12,16,3,1),#10x10x16
            nn.MaxPool2d(2,2),# 5x5x16
            nn.Conv2d(16,32,3,2), # 2x2x32
            #nn.MaxPool2d(2,1) # 1x1x32
            )

        self.decoder = nn.Sequential( #TODO : decoder architecture
            nn.ConvTranspose2d(32, 16, 3, stride=1),  # 3x3x16
            nn.ConvTranspose2d(16, 8, 5, 2, padding=1), # 7x7x8
            nn.ConvTranspose2d(8, 4, 3, 2),  # 15x15x4
            nn.ConvTranspose2d(4, 1, 3, 2, output_padding=1),  # 32x32x1
            nn.Sigmoid()
            )

    def forward(self, x):
        en = self.encoder(x)
        out  = self.decoder(en)
        #print(out.data.shape)
        #c_out= out.cpu().data.numpy()
        #c_x =  x.cpu().data.numpy()

        #crop the centeral box with the same size as input.

        x_off = int((out.size()[2]- x.size()[2])/2)
        y_off = int((out.size()[3]- x.size()[3])/2)
        out = out[:,:,x_off:x_off+ x.size()[2], y_off:y_off+x.size()[3]]

        return out

    def get_features(self, x):
        en = self.encoder(x)
        en  = en.view(-1, 32)
        return en
