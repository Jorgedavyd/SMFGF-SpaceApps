import torch.nn as nn

# Base Deep Neural Network
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, *args, activation=None):
        super(DeepNeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(nn.Linear(input_size, output_size))
        if activation is not None:
            self.output_layer.add_module(activation)
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out
    
# Attention based RNNs
class Attention(nn.Module):
    def __init__(self, input_size, num_heads = 1):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(input_size, num_heads, batch_first = True)
        self.layer_norm = nn.LayerNorm(input_size)
    
    def forward(self, x):
        
        attn, _ = self.attn(x,x,x)
        context = self.layer_norm(attn)
        return context