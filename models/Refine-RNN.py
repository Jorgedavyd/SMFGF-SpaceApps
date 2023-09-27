from models.rnn_models import *


class Refine_RNN(RegBase):
    def __init__(self, encoder, fc):
        self.encoder = encoder
        self.fc = fc   #RNN with input size: hidden_size, hidden_size: hidden_size, the encoder will encode min into 1/3 time step and 1/3 of features.
    def forward(self, x):
        with torch.no_grad():
            out = self.encoder(x) #We will train it apart
        out = self.fc(out)
        return out
    
class Refine_Encoder(RegBase):
    def __init__(self, encoder):
        self.encoder = encoder
    def forward(self, x):
        return self.encoder(x)