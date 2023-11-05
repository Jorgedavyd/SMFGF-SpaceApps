from models.mult2multatt import *
from models.utils import *
from models.base import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
class GeoVideoCNNRNN(GeoBase):
    def __init__(self, image_model, rnn):
        super(GeoVideoCNNRNN, self).__init__()
        self.feature_extractor = image_model
        self.sequential_extractor = rnn
    def forward(self, x):
        _, seq_length, _,_,_ = x.size()
        feature_extraction = []
        
        #feature extraction
        for t in range(seq_length):
            feature_extraction.append(self.feature_extractor(x[:,t,:,:,:]))
        
        out = torch.cat(feature_extraction)
        #sequential analysis
        if isinstance(self.sequential_extractor, ResidualMultiheadAttentionLSTM):
            _, (hn,_) = self.sequential_extractor(out)
        else:
            _, hn = self.sequential_extractor(out)
        
        return hn
class ThreeD_CNN(nn.Module):
    def __init__(self, hidden_state_size, architecture):
        from torchvision.models.video import mc3_18, MC3_18_Weights
        super(ThreeD_CNN, self).__init__()
        
        self.model = mc3_18(weights = MC3_18_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = DeepNeuralNetwork(512, hidden_state_size, *architecture)

        self.transform = tt.Compose([tt.ToTensor(), MC3_18_Weights.KINETICS400_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn
