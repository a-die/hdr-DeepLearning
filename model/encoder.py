# -- coding: utf-8 --
# @Time : 2022/11/4 10:43
# @Author : 欧阳亨杰
# @File : encoder.py
import numpy as np
import timm
import torch
from torch import nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        #         self.densenet = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.cnn.fc.in_features
        self.cnn.global_pool = nn.Identity()
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        bs = x.size(0)
        features = self.cnn(x)
        features = features.permute(0, 2, 3, 1)
        return features


if __name__ == '__main__':
    encoder = Encoder(model_name='resnet18')
    print(encoder)
    img = torch.from_numpy(np.ones((8, 3, 224, 224))).float()
    res = encoder(img)
    print(res.shape)
