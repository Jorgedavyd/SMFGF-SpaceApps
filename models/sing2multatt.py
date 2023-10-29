import torch.nn as nn
import torch.nn.functional as F
from models.macro_architectures import get_lr
from df_models import *
from models.base import *
import torch
#not multihead to multiheadattention
class SingleHead2MultiHead(nn.Module):
    def training_step(self, batch, weights:list, encoder_forcing):
        self.encoder_forcing = encoder_forcing
        if encoder_forcing:
            if isinstance(self, LSTMSeq2Seq):
                #instantiate loss
                loss = 0
                alpha_encoder, alpha_out, alpha_main = weights
                l1_data, l2_data, target = batch #decompose batch
                
                out_L1, (hn_L1, cn_L1) = self.lstm_encoder(l1_data)
                out_L2, (hn_L2, cn_L2) = self.lstm_encoder(l2_data)
                loss += F.mse_loss(out_L1, out_L2)*alpha_encoder
                ctx_L1, _ = self.multatt(out_L1,out_L1, out_L1)
                out_L1 = self.layernorm(ctx_L1+out_L1)
                ctx_L2, _ = self.multatt(out_L2,out_L2, out_L2)
                out_L2 = self.layernorm(ctx_L2+out_L2)
                out_L1, (_,_) = self.lstm_decoder(out_L1, (hn_L1, cn_L1))
                out_L2, (_,_) = self.lstm_decoder(out_L2, (hn_L2, cn_L2))
                loss += F.mse_loss(out_L1, out_L2)*alpha_out
                out_L1 = out_L1[:, -1, :]
                out_L1 = self.fc_decoder(out_L1)
                loss += F.mse_loss(out_L1, target)*alpha_main
            elif isinstance(self, GRUSeq2Seq):
                #instantiate loss
                loss = 0
                alpha_encoder, alpha_out, alpha_main = weights
                l1_data, l2_data, target = batch #decompose batch
                
                out_L1, hn_L1 = self.gru_encoder(l1_data)
                out_L2, hn_L2 = self.gru_encoder(l2_data)
                loss += F.mse_loss(out_L1, out_L2)*alpha_encoder
                ctx_L1, _ = self.multatt(out_L1,out_L1, out_L1)
                out_L1 = self.layernorm(ctx_L1+out_L1)
                ctx_L2, _ = self.multatt(out_L2,out_L2, out_L2)
                out_L2 = self.layernorm(ctx_L2+out_L2)
                out_L1, _ = self.gru_decoder(out_L1, hn_L1)
                out_L2, _ = self.gru_decoder(out_L2, hn_L2)
                loss += F.mse_loss(out_L1, out_L2)*alpha_out
                out_L1 = self.fc_decoder(out_L1)
                out_L1 = out_L1[:, -1, :]
                loss += F.mse_loss(out_L1, target)*alpha_main
        else:
            l1_data, target = batch
            out = self(l1_data)
            loss = F.mse_loss(out, target)
        return loss
    def validation_step(self, batch):
        if self.encoder_forcing:
            l1_data,_, target = batch #decompose batch
        else:
            l1_data, target = batch 
        pred = self(l1_data)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        r2_ = r2(pred, target)
        precision, recall, f1_score = compute_all(pred, target)
        return {'val_loss': loss.detach(),'r2': r2_.detach(), 'precision': precision, 'recall': recall, 'f1': f1_score}
class LSTMSeq2Seq(SingleHead2MultiHead):
    def __init__(self, input_size, output_size, hidden_size, num_heads, arquitecture, num_layers = 1, dropout = 0, bidirectional = True): #input 20, #hidden 10, #pred_length with transformation
        super(LSTMSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)

        ## Attention mechanism 
        self.multatt = nn.MultiheadAttention(hidden_size, num_heads, batch_first = True)
        self.layernorm = nn.LayerNorm(hidden_size)
        # Decoder
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True) #encoders[0].hidden_size*len(encoders) hidden_sizeto hidden_size
        self.fc_decoder = DeepNeuralNetwork(hidden_size, output_size, *arquitecture)

    def forward(self, x):
        out, (hn,cn) = self.lstm_encoder(x)
        att, _ = self.multatt(out,out,out)
        att = self.layernorm(att)
        out, (_,_) = self.lstm_decoder(att, (hn,cn))
        out = out[:,-1, :]
        out = self.fc_decoder(out)
        return out

##Seq2Seq models with attention
class GRUSeq2Seq(SingleHead2MultiHead):
    def __init__(self, input_size, output_size, hidden_size, num_heads, arquitecture, num_layers = 1, dropout = 0, bidirectional = True): #input 20, #hidden 10, #pred_length with transformation
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru_encoder = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)

        ## Attention mechanism 
        self.multatt = nn.MultiheadAttention(hidden_size, num_heads, batch_first = True)
        self.layernorm = nn.LayerNorm(hidden_size)
        # Decoder
        self.gru_decoder = nn.GRU(hidden_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True) #encoders[0].hidden_size*len(encoders) hidden_sizeto hidden_size
        self.fc_decoder = DeepNeuralNetwork(hidden_size, output_size, *arquitecture)

    def forward(self, x):
        out, hn = self.gru_encoder(x)
        att, _ = self.multatt(out,out,out)
        att = self.layernorm(att)
        out, _ = self.gru_decoder(att, hn)
        out = out[:,-1, :]
        out = self.fc_decoder(out)
        return out