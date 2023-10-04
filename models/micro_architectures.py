import torch.nn as nn
import torch
    
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

class Simple1DCNN(nn.Module):
    def __init__(self,architecture, input_size, hidden_size, kernel_size=3, stride=2):
        super(Simple1DCNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv1d = nn.Conv1d(input_size, 10, kernel_size, stride)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = DeepNeuralNetwork(40, hidden_size,*architecture)
        ##add attention
    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class DeepVanillaRNN(nn.Module):
    def __init__(self, hidden_size, input_size, batch_size, mlp_architecture, hidden_state = None):
        super(DeepVanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        if hidden_state is None:
            self.hidden_state = torch.zeros(1, batch_size, hidden_size, requires_grad=True)
        else:
            self.hidden_state = hidden_state
        self.hidden_mlp = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.input_mlp = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        ##add attention
    def forward(self,x):
        a_t = self.hidden_mlp(self.hidden_state) + self.input_mlp(x)
        self.hidden_state = torch.tanh(a_t)
        return self.hidden_state

class DeepLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, batch_size, mlp_architecture, hidden_state = None, cell_state = None):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        #Forget gate
        self.F_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.F_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Input gate
        self.I_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.I_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Ouput gate
        self.O_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.O_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Input node
        self.C_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.C_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)        
        #c and H
        if hidden_state is None:
            self.H = torch.zeros(1, batch_size, hidden_size, requires_grad=True)
        else:
            self.H = hidden_state
        if cell_state is None:
            self.C = torch.zeros(1, batch_size, hidden_size, requires_grad = True)
        else:
            self.C = cell_state
        #add attention
    def forward(self,x):
        self.a_F = self.F_h(self.H) + self.F_x(x)
        self.F = torch.sigmoid(self.a_F)
        self.a_I = self.I_h(self.H) + self.I_x(x)
        self.I = torch.sigmoid(self.a_I)
        self.a_O = self.O_h(self.H) + self.O_x(x)
        self.O = torch.sigmoid(self.a_O)
        self.a_C_hat = self.C_hat_h(self.H) + self.C_hat_x(x)
        self.C_hat = torch.tanh(self.a_C_hat)
        self.C = self.F*self.C + self.I*self.C_hat
        self.H = self.O*torch.tanh(self.C)
        return self.H

class DeepGRU(nn.Module):
    def __init__(self, hidden_size, input_size, batch_size, mlp_architecture, hidden_state = None):
        super(DeepGRU, self).__init__()
        self.hidden_size = hidden_size
        #Update gate
        self.Z_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.Z_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Reset gate
        self.R_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.R_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Possible hidden state
        self.H_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.H_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        if hidden_state is None:
            self.H = torch.zeros(1, batch_size, hidden_size, requires_grad=True)
        else:
            self.H = hidden_state
        #add attention
    def forward(self, x):
        self.Z = torch.sigmoid(self.Z_h(self.H)+self.Z_x(x))
        self.R = torch.sigmoid(self.R_h(self.H)+self.R_x(x))
        self.H_hat = torch.tanh(self.H_hat_h(self.H*self.R)+self.H_hat_x(x))
        self.H = self.H*self.Z + (torch.ones_like(self.Z)-self.Z)*self.H_hat
        return self.H
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        _, (h0,_) = self.lstm(x)
        return h0
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        _, h0 = self.gru(x) 
        return h0
    
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
    
    def forward(self, x, h0):
        _, hn = self.rnn(x, h0)
        return hn  
    
class BidirectionalRNNWithAttention(nn.Module):
    def __init__(self, rnn1, rnn2):
        super(BidirectionalRNNWithAttention, self).__init__()
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        # Attention layers
        self.attention_layer = nn.Linear(2 * self.rnn1.hidden_size, 2*self.rnn1.hidden_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Forward pass through the first RNN
        hidden1 = self.rnn1(x)
        
        # Reverse the input sequence for the second RNN
        x_backward = torch.flip(x, [1])
        
        # Forward pass through the second RNN
        hidden2 = self.rnn2(x_backward)
        
        # Concatenate to bidirectional output
        hidden_bidirectional = torch.cat([hidden1,hidden2], dim = 2)
        
        # Compute the scores
        attn_scores = self.attention_layer(hidden_bidirectional)

        #Compute the attention weights

        attn_weights = self.softmax(attn_scores)

        #Compute the context

        context = torch.sum(attn_weights*hidden_bidirectional, dim = 1)
        
        return context

#Create non deep ones