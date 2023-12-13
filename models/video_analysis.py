from models.mult2multatt import *
from models.utils import *
from models.base import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as tt

class GeoVideoCNNRNN(GeoBase):
    def __init__(self, image_model, rnn, architecture, dropout = 0.2):
        super(GeoVideoCNNRNN, self).__init__()
        self.feature_extractor = image_model
        self.sequential_extractor = rnn
        self.fc = nn.Sequential(
            DeepNeuralNetwork(rnn.hidden_size, rnn.hidden_size/2, *architecture),
            nn.Dropout(dropout, inplace = True)
            )
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
class MC3_18(nn.Module):
    def __init__(self, hidden_state_size, architecture, dropout: float = 0.2):
        from torchvision.models.video import mc3_18, MC3_18_Weights
        super(MC3_18, self).__init__()
        
        self.model = mc3_18(weights = MC3_18_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            DeepNeuralNetwork(512, hidden_state_size, *architecture),
            nn.Dropout(dropout, inplace = True)
            )

        self.transform = tt.Compose([tt.ToTensor(), MC3_18_Weights.KINETICS400_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn
    

class MVIT_V2_S(nn.Module):
    def __init__(self, hidden_state_size, architecture, dropout: float = 0.2):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        super(MVIT_V2_S, self).__init__()
        
        self.model = mvit_v2_s(weights = MViT_V2_S_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Sequential(
            DeepNeuralNetwork(768, hidden_state_size, *architecture),
            nn.Dropout(dropout, inplace = True)
            )

        self.transform = tt.Compose([tt.ToTensor(), MViT_V2_S_Weights.KINETICS400_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn

class SWIN3D_B(nn.Module):
    def __init__(self, hidden_state_size, architecture, dropout: float = 0.2):
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights
        super(SWIN3D_B, self).__init__()
        
        self.model = swin3d_b(weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Sequential(
            DeepNeuralNetwork(1024, hidden_state_size, *architecture),
            nn.Dropout(dropout, inplace = True)
            )

        self.transform = tt.Compose([tt.ToTensor(), Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms(antialias=True)])
    def forward(self, x):
        hn = self.model(x)
        return hn