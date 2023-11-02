import torch
import torch.nn as nn
import torch.nn.functional as F
from base import *
from models.utils import DeepNeuralNetwork

class DAE(GeoBase):
    def __init__(self, task):
        GeoBase.__init__(self, task)
    def training_step(self, batch):
        l1, l2 = batch
        l2_hat = self(l1)
        loss = F.mse_loss(l2_hat, l2)
        l2_hat = self(0.2*torch.randn(l2_hat.size()) * l2)
        loss += F.mse_loss(l2_hat, l2)
        return loss
    
    def validation_step(self, batch):
        l1, l2 = batch
        l2_hat = self(l1)
        loss = F.mse_loss(l2_hat, l2)
        l2_hat = self(0.2*torch.randn(l2_hat.size()) * l2)
        loss += F.mse_loss(l2_hat, l2)
        return loss


class LSTMDenoisingAutoEncoder(DAE):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, num_heads = None, architecture = (20,20,20)):
        super(LSTMDenoisingAutoEncoder, self).__init__()
        DAE.__init__(self, 'regression')
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.attention = True if num_heads is not None else False
        if num_heads is not None:
            self.atn_encoder = nn.MultiheadAttention(input_size, num_heads[0])
            self.atn_context = nn.MultiheadAttention(hidden_size*2 if bidirectional else hidden_size, num_heads[1])
            self.layer_norm_1 = nn.LayerNorm(input_size)
            self.layer_norm_2 = nn.LayerNorm(hidden_size*2 if bidirectional else hidden_size)
        self.fc = DeepNeuralNetwork(input_size, input_size, *architecture)
    def forward(self, x):
        _, seq_length, _ = x.size()
        #encoder
        attn, _ = self.atn_encoder(x,x,x)
        attn = self.layer_norm_1(attn)
        out, (_,_) = self.encoder(attn)

        #context vector manipulation
        attn, _ = self.atn_context(out, out, out)
        attn = self.layer_norm_2(attn)
        #decoder output
        out, (_,_) = self.decoder(attn)
        out_list = []
        for t in range(seq_length):
            xt = out[:,t,:]
            pred = self.fc(xt)
            out_list.append(pred)
        out = torch.cat(out_list, dim = 1)
        return out

class GRUDenoisingAutoEncoder(DAE):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, num_heads = None, architecture = (20,20,20)):
        super(GRUDenoisingAutoEncoder, self).__init__()
        DAE.__init__(self, 'regression')
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.decoder = nn.GRU(hidden_size, input_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.attention = True if num_heads is not None else False
        if num_heads is not None:
            self.atn_encoder = nn.MultiheadAttention(input_size, num_heads[0])
            self.atn_context = nn.MultiheadAttention(hidden_size*2 if bidirectional else hidden_size , num_heads[1])
            self.layer_norm_1 = nn.LayerNorm(input_size)
            self.layer_norm_2 = nn.LayerNorm(hidden_size*2 if bidirectional else hidden_size)
        self.fc = DeepNeuralNetwork(input_size, input_size, *architecture)
    def forward(self, x):
        _, seq_length, _ = x.size()
        #encoder
        attn, _ = self.atn_encoder(x,x,x)
        attn = self.layer_norm_1(attn)
        out, _ = self.encoder(attn)

        #context vector manipulation
        attn, _ = self.atn_context(out, out, out)
        attn = self.layer_norm_2(attn)
        #decoder output
        out, _ = self.decoder(attn)
        out_list = []
        for t in range(seq_length):
            xt = out[:,t,:]
            pred = self.fc(xt)
            out_list.append(pred)
        out = torch.cat(out_list, dim = 1)
        return out



        
        

