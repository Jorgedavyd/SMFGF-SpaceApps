import torch.nn as nn
import torch
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        attention_weights = self.attention_layer(hidden_states)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        context_vector = torch.sum(attention_weights * hidden_states, dim=0)
        
        return context_vector
    
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
    def __init__(self, hidden_size, input_size, mlp_architecture):
        super(DeepVanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_mlp = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.input_mlp = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        self.attention = Attention(self.hidden_size)
        ##add attention
    def forward(self,x):
        batch_size, seq_len, _ = x.size()

        hn = torch.zeros(batch_size, self.hidden_size, requires_grad=True)
        hn_list = []

        #create here loop for training the entire sequence
        for t in range(seq_len):
            xt = x[:, t, :]# Extract the input at time t
            
            a_t = self.hidden_mlp(hn) + self.input_mlp(xt)
            
            hn = torch.tanh(a_t)

            hn_list.append(hn)
        hidden_states = torch.stack(hn_list)
        context = self.attention(hidden_states)
        return context

class DeepLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture):
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
        self.attention = Attention(self.hidden_size)      
    def forward(self,x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []

        for t in range(sequence_size):
            xt = x[:, t, :]
            #forward
            a_F = self.F_h(hn) + self.F_x(xt)
            F = torch.sigmoid(a_F) #forget gate
            a_I = self.I_h(hn) + self.I_x(xt)
            I = torch.sigmoid(a_I) #input gate
            a_O = self.O_h(hn) + self.O_x(xt)
            O = torch.sigmoid(a_O) #output gate
            a_C_hat = self.C_hat_h(hn) + self.C_hat_x(xt)
            C_hat = torch.tanh(a_C_hat)
            cn = F*cn + I*C_hat
            hn = O*torch.tanh(cn)
            hn_list.append(hn)

        hidden_states = torch.stack(hn_list)
        context = self.attention(hidden_states)
        
        return context

class DeepGRU(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture):
        super(DeepGRU, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(self.hidden_size)
        #Update gate
        self.Z_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.Z_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Reset gate
        self.R_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.R_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Possible hidden state
        self.H_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.H_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            xt = x[:, t, :]
            Z = torch.sigmoid(self.Z_h(hn)+self.Z_x(xt))
            R = torch.sigmoid(self.R_h(hn)+self.R_x(xt))
            H_hat = torch.tanh(self.H_hat_h(hn*R)+self.H_hat_x(xt))
            hn = hn*Z + (torch.ones_like(Z)-Z)*H_hat
            hn_list.append(hn)
        hidden_states = torch.stack(hn_list)
        context = self.attention(hidden_states)
        return context
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.attention = Attention(self.hidden_size)

    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn,cn = self.lstm(xt, (hn,cn))
            hn_list.append(hn)
        hidden_states = torch.stack(hn_list)
        context = self.attention(hidden_states)

        return context
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.attention = Attention(self.hidden_size)

    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn = self.gru(xt, hn) 

            hn_list.append(hn)
        
        hidden_states = torch.stack(hn_list)
        
        context = self.attention(hidden_states)
        return context
    
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.attention = Attention(self.hidden_size)
    
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn = self.rnn(xt, hn)

            hn_list.append(hn)

        hidden_states = torch.stack(hn_list)

        context_vector = self.attention(hidden_states)
        
        return context_vector
    
class BidirectionalRNNWithAttention(nn.Module):
    def __init__(self, rnn1, rnn2):
        super(BidirectionalRNNWithAttention, self).__init__()
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        
    def forward(self, x):
        # Forward pass through the first RNN
        hidden1 = self.rnn1(x)
        # Reverse the input sequence for the second RNN
        x_backward = torch.flip(x, [1])
        
        # Forward pass through the second RNN
        hidden2 = self.rnn2(x_backward)

        # Concatenate to bidirectional output
        hidden_bidirectional = torch.cat((hidden1,hidden2), dim = 1)
        
        return hidden_bidirectional

