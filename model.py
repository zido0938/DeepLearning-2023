
import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(CustomResNet, self).__init__()
        resnet = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        resnet.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.BatchNorm1d(256),  # Add Batch Normalization
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)