import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import *
from models.utils import DeepNeuralNetwork

class DAE(GeoBase):
    def __init__(self):
        super(DAE, self).__init__('regression')
    def training_step(self, batch,  weights = None, encoder_forcing = None):
        l1, l2 = batch
        l2_hat = self(l1)
        loss = F.mse_loss(l2_hat, l2)
        return loss
    
    def validation_step(self, batch):
        l1, l2 = batch
        l2_hat = self(l1)
        loss = F.mse_loss(l2_hat, l2)
        r2_ = r2(l2_hat, l2)
        return {'val_loss': loss.detach(), 'r2': r2_.detach()}

class TransformerDenoisingAutoencoder(DAE):
    def __init__(self, encoder, decoder):
        super(TransformerDenoisingAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        return output
    
class LSTMDenoisingAutoEncoder(DAE):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, num_heads = None, architecture = (20,20,20)):
        super(LSTMDenoisingAutoEncoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.decoder = nn.LSTM(hidden_size*2 if bidirectional else hidden_size, input_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.attention = True if num_heads is not None else False
        if num_heads is not None:
            self.atn_encoder = nn.MultiheadAttention(input_size, num_heads[0])
            self.atn_context = nn.MultiheadAttention(hidden_size*2 if bidirectional else hidden_size, num_heads[1])
            self.layer_norm_1 = nn.LayerNorm(input_size)
            self.layer_norm_2 = nn.LayerNorm(hidden_size*2 if bidirectional else hidden_size)
        self.fc = DeepNeuralNetwork(input_size*2 if bidirectional else input_size, input_size, *architecture)
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
        out = out.view(-1, seq_length, self.input_size)
        return out

class GRUDenoisingAutoEncoder(DAE):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, num_heads = None, architecture = (20,20,20)):
        super(GRUDenoisingAutoEncoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.decoder = nn.GRU(hidden_size*2 if bidirectional else hidden_size, input_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.attention = True if num_heads is not None else False
        if num_heads is not None:
            self.atn_encoder = nn.MultiheadAttention(input_size, num_heads[0])
            self.atn_context = nn.MultiheadAttention(hidden_size*2 if bidirectional else hidden_size , num_heads[1])
            self.layer_norm_1 = nn.LayerNorm(input_size)
            self.layer_norm_2 = nn.LayerNorm(hidden_size*2 if bidirectional else hidden_size)
        self.fc = DeepNeuralNetwork(input_size*2 if bidirectional else input_size, input_size, *architecture)
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
        out = out.view(-1, seq_length, self.input_size)
        return out



        
        

