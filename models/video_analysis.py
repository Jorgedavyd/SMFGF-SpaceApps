from models.mult2multatt import *
from models.utils import *
from models.base import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class GeoVideo(nn.Module):
    def __init__(self, image_model, rnn):
        super().__init__()
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