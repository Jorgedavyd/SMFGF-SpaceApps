from models.macro_architectures import *
import torch.nn as nn
import torch

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
    


class Simple1DCNN(nn.Module):
    def __init__(self,architecture, input_size, hidden_size, num_heads = None, kernel_size=3, stride=2):
        super(Simple1DCNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv1d = nn.Conv1d(input_size, 10, kernel_size, stride)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = DeepNeuralNetwork(40, hidden_size,*architecture)
        ##add attention
        self.num_heads = num_heads
        if num_heads is not None:
            self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
    def forward(self, x):
        if self.num_heads is not None:
            x,_ = self.attention(x,x,x)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class DeepVanillaRNN(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture, attention):
        super(DeepVanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.at = attention
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
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn

class DeepLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture, attention):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.at = attention
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

        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn

class DeepGRU(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture, attention):
        super(DeepGRU, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(self.hidden_size)
        self.at = attention
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
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, attention = True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.attention = Attention(self.hidden_size)
        self.at = attention
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn,cn = self.lstm(xt, (hn,cn))
            hn_list.append(hn)
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, attention):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.attention = Attention(self.hidden_size)
        self.at = attention
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn = self.gru(xt, hn) 

            hn_list.append(hn)
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
    
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, attention = True):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.attention = Attention(self.hidden_size)
        self.at = attention
    
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn = self.rnn(xt, hn)

            hn_list.append(hn)

        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
    
class BidirectionalRNN(nn.Module):
    def __init__(self, rnn1, rnn2):
        super(BidirectionalRNN, self).__init__()
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

#training for seq2seq with feature separation
class MultiHead2MultiHeadBase(nn.Module):
    def training_step(self, batch):
        fc, mg, target = batch #decompose batch
        pred = self(fc, mg)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch):
        fc, mg, target = batch #decompose batch
        pred = self(fc, mg)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, max_lr, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento

        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        # Learning rate scheduler, le da momento inicial al entrenamiento para converger con valores menores al final
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            lrs = [] # Seguimiento
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch)
                #Seguimiento
                train_losses.append(loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()

                # Guardar el learning rate utilizado en el cycle.
                lrs.append(get_lr(optimizer))
                #Utilizar el siguiente valor de learning rate dado OneCycle scheduler
                sched.step()

            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            result['lrs'] = lrs #Guarda la lista de learning rates de cada batch
            self.epoch_end(epoch, result) #imprimir en pantalla el seguimiento
            history.append(result) # añadir a la lista el diccionario de resultados
        return history
#LSTMs with multihead attention incorporated
class EncoderMultiheadAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, architecture):
        super(EncoderMultiheadAttentionLSTM, self).__init__()
        self.hidden_size=hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        #encoder
        self.lstm = nn.LSTMCell(input_size,hidden_size)
        self.fc = DeepNeuralNetwork(hidden_size,hidden_size, *architecture)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        #Multihead attention
        attn_out, _ = self.attention(x,x,x)
        #residual connection and layer_norm
        attn_out = self.layer_norm(attn_out+x)
        
        #encoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn,cn = self.lstm(xt, (hn,cn))
            out = self.fc(hn)
            out_list.append(out)
        
        out = torch.stack(out_list, dim = 1)
        
        out = self.layer_norm(out+attn_out)

        return out, (hn, cn)
    
class MultiHeaded2MultiheadAttentionLSTM(MultiHead2MultiHeadBase):
    def __init__(self, encoder_fc, encoder_mg,num_heads, architecture, output_size):
        super(MultiHeaded2MultiheadAttentionLSTM, self).__init__()
        #hidden
        self.hidden_size = encoder_fc.hidden_size + encoder_mg.hidden_size
        #encoder(LSTMWithMultiHeadAttention)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        #MultiheadAttention
        self.attention_1 = nn.MultiheadAttention(encoder_fc.hidden_size, num_heads, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(encoder_mg.hidden_size, num_heads, batch_first=True)
        #Decoder arch
        self.lstm_1 = nn.LSTMCell(encoder_fc.hidden_size, encoder_fc.hidden_size)
        self.lstm_2 = nn.LSTMCell(encoder_fc.hidden_size, encoder_fc.hidden_size)
        self.linear_1 = DeepNeuralNetwork(encoder_fc.hidden_size, encoder_fc.hidden_size, *architecture)
        self.lstm_3 = nn.LSTMCell(encoder_mg.hidden_size, encoder_mg.hidden_size)
        self.lstm_4 = nn.LSTMCell(encoder_mg.hidden_size, encoder_mg.hidden_size)
        self.linear_2 = DeepNeuralNetwork(encoder_mg.hidden_size, encoder_mg.hidden_size, *architecture)
        self.fc = DeepNeuralNetwork(self.hidden_size, output_size, *architecture)
        #layer norm with residual connections(AttentionIsAllYouNeed uses several times on arch)
        self.layer_norm_1 = nn.LayerNorm(encoder_fc.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(encoder_mg.hidden_size)
    def forward(self, fc, mg):
        hn_list = []
        #get dim
        _, seq_length, _ = fc.size()
        #encoder for faraday cup
        out, (hn,cn) = self.encoder_fc(fc)
        #Attention mechanism
        attn_out, _ = self.attention_1(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_1(attn_out+fc)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn,cn = self.lstm_1(xt, (hn,cn))
            out = self.linear_1(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_1(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn,cn = self.lstm_2(xt, (hn,cn))
        #add to last hn 
        hn_list.append(hn)
        #encoder for magnetometer
        out, (hn,cn) = self.encoder_mg(mg)
        #Attention mechanism
        attn_out, _ = self.attention_2(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_2(attn_out+mg)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn,cn = self.lstm_3(xt, (hn,cn))
            out = self.linear_2(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_2(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn,cn = self.lstm_4(xt, (hn,cn))
        hn_list.append(hn)
        
        hn = torch.cat(hn_list, dim = 1)
        #inference with Deep Neural Network

        out = self.fc(hn)
        return out
#GRUs with multihead attention incorporated
class EncoderMultiheadAttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, architecture):
        super(EncoderMultiheadAttentionGRU, self).__init__()
        self.hidden_size=hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first = True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        #encoder
        self.gru = nn.GRUCell(input_size,hidden_size)
        self.fc = DeepNeuralNetwork(hidden_size,hidden_size, *architecture)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        #Multihead attention
        attn_out, _ = self.attention(x,x,x)
        #residual connection and layer_norm
        attn_out = self.layer_norm(attn_out+x)
        
        #encoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn = self.gru(xt, hn)
            out = self.fc(hn)
            out_list.append(out)
        
        out = torch.stack(out_list, dim = 1)
        
        out = self.layer_norm(out+attn_out)

        return out, hn
class MultiHeaded2MultiheadAttentionGRU(MultiHead2MultiHeadBase):
    def __init__(self, encoder_fc, encoder_mg,num_heads, architecture, output_size):
        super(MultiHeaded2MultiheadAttentionGRU, self).__init__()
        #hidden
        self.hidden_size = encoder_fc.hidden_size + encoder_mg.hidden_size
        #encoder(LSTMWithMultiHeadAttention)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        #MultiheadAttention
        self.attention_1 = nn.MultiheadAttention(encoder_fc.hidden_size, num_heads, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(self.hidden_size, num_heads, batch_first=True)
        #Decoder arch
        self.gru_1 = nn.GRUCell(encoder_fc.hidden_size, encoder_fc.hidden_size)
        self.gru_2 = nn.GRUCell(encoder_fc.hidden_size, encoder_fc.hidden_size)
        self.linear_1 = DeepNeuralNetwork(encoder_fc.hidden_size, encoder_fc.hidden_size, *architecture)
        self.gru_3 = nn.GRUCell(encoder_mg.hidden_size, encoder_mg.hidden_size)
        self.gru_4 = nn.GRUCell(encoder_mg.hidden_size, encoder_mg.hidden_size)
        self.linear_2 = DeepNeuralNetwork(encoder_mg.hidden_size, encoder_mg.hidden_size, *architecture)
        self.fc = DeepNeuralNetwork(self.hidden_size, output_size, *architecture)
        #layer norm with residual connections(AttentionIsAllYouNeed uses several times on arch)
        self.layer_norm_1 = nn.LayerNorm(encoder_fc.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(encoder_mg.hidden_size)
    def forward(self, fc, mg):
        hn_list = []
        #get dim
        _, seq_length, _ = fc.size()
        #encoder for faraday cup
        out, hn = self.encoder_fc(fc)
        #Attention mechanism
        attn_out, _ = self.attention_1(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_1(attn_out+fc)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn = self.gru_1(xt, hn)
            out = self.linear_1(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_1(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn = self.gru_2(xt, hn)
        #add to last hn 
        hn_list.append(hn)
        #encoder for magnetometer
        out, hn = self.encoder_mg(mg)
        #Attention mechanism
        attn_out, _ = self.attention_2(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_2(attn_out+mg)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn = self.gru_3(xt, hn)
            out = self.linear_2(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_2(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn = self.gru_4(xt, hn)
        hn_list.append(hn)
        
        hn = torch.cat(hn_list, dim = 1)
        #inference with Deep Neural Network

        out = self.fc(hn)
        return out
#not multihead to multiheadattention
class SingleHead2MultiHead(nn.Module):
    def training_step(self, batch):
        feature, target = batch #decompose batch
        pred = self(feature)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch):
        feature, target = batch #decompose batch
        pred = self(feature)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, max_lr, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento

        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        # Learning rate scheduler, le da momento inicial al entrenamiento para converger con valores menores al final
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            lrs = [] # Seguimiento
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch)
                #Seguimiento
                train_losses.append(loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()

                # Guardar el learning rate utilizado en el cycle.
                lrs.append(get_lr(optimizer))
                #Utilizar el siguiente valor de learning rate dado OneCycle scheduler
                sched.step()

            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            result['lrs'] = lrs #Guarda la lista de learning rates de cada batch
            self.epoch_end(epoch, result) #imprimir en pantalla el seguimiento
            history.append(result) # añadir a la lista el diccionario de resultados
        return history
class Seq2SeqLSTM(SingleHead2MultiHead):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTMCell(input_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, input_size)

        ## Attention mechanism 
        self.attention_2 = nn.MultiheadAttention(input_size, num_heads, batch_first = True)
        # Decoder

        self.lstm_2 = nn.LSTMCell(input_size, hidden_size) #encoders[0].hidden_size*len(encoders) hidden_sizeto hidden_size
        self.fc_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        out_list = []
        for i in range(seq_length):
            xt = x[:,i,:]
            hn,cn = self.lstm_1(xt, (hn,cn))
            out = self.fc_1(hn)
            out_list.append(out)
        
        query = torch.stack(out_list, dim=1)

        # Multihead Attention
        attention_output, _ = self.attention_2(query, query, query)

        # Decoding
        for i in range(seq_length):
            xt = attention_output[:,i,:]
            hn,cn = self.lstm_2(xt, (hn,cn))
        
        out = self.fc_2(hn)

        return out

##Seq2Seq models with attention
class GRUSeq2Seq(SingleHead2MultiHead):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(GRUSeq2Seq, self).__init__()
        self.gru_1 = nn.LSTMCell(input_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, input_size)

        ## Attention mechanism 
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first = True)
        # Decoder

        self.gru_2 = nn.LSTMCell(input_size, hidden_size) #encoders[0].hidden_size*len(encoders) hidden_sizeto hidden_size
        self.fc_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        out_list = []
        for i in range(seq_length):
            xt = x[:,i,:]
            hn,cn = self.gru_1(xt, (hn,cn))
            out = self.fc_1(hn)
            out_list.append(out)
        
        query = torch.stack(out_list, dim=1)

        # Multihead Attention
        attention_output, _ = self.attention(query, query, query)

        # Decoding
        for i in range(seq_length):
            xt = attention_output[:,i,:]
            hn,cn = self.gru_2(xt, (hn,cn))
        
        out = self.fc_2(hn)

        return out

