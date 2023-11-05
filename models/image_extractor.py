import torchvision.transforms as tt
from models.utils import *
import torch.nn as nn

class FeatureExtractorResnet50(nn.Module):
    def __init__(self, architecture):
        super(FeatureExtractorResnet50, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = DeepNeuralNetwork(1280, 100, *architecture)
        
        self.transform = tt.Compose([tt.ToTensor(), ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)])
    def forward(self, x):
        out = self.model(x)
        return out

class FeatureExtractorVGG19(nn.Module):
    def __init__(self, architecture):
        super(FeatureExtractorVGG19, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights

        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = DeepNeuralNetwork(1280, 100, *architecture)
        
        self.transform = tt.Compose([tt.ToTensor(), VGG19_Weights.IMAGENET1K_V1.transforms(antialias=True)])
    def forward(self, x):
        out = self.model(x)
        return out