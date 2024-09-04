# import torch
from torch import nn
from torchvision.models import vgg
import config

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)