import torch
import torch.nn as nn
import math
from models.base import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)  # Adjust input size to match satellite_size
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        return memory
    
class TransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = PositionalEncoding(hidden_size, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers,
            batch_first = True
        )

        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        output = self.output_projection(output)
        return output

class TransformerMultiEncoder2Decoder(GeoBase):
    def __init__(self, encoders, decoder):
        super(TransformerMultiEncoder2Decoder, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoder = decoder
    def forward(self, trg, src, tgt_mask = None, src_mask = None):
        memory_list = []
        for encoder, input in zip(self.encoders, src):
            memory_list.append(encoder(input))

        memory = torch.cat(memory_list, dim = -1)

        output = self.decoder(trg, memory, tgt_mask, src_mask)

        return output
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x, y) 
        J = F.mse_loss(y_hat, y)
        return J
    @torch.no_grad()
    def validation_step(self, batch):
        x,y = batch
        y_hat = self(x)
        r2_score = r2(y_hat, y)
        val_loss = F.mse_loss(y_hat, y)
        return {'val_loss': val_loss, 'r2': r2_score}
    
