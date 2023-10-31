import torch
import torch.nn as nn
import torch.nn.functional as F
from base import *
from models.utils import DeepNeuralNetwork

class DAE(GeoBase):
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
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))
    
    def epoch_end_one_cycle(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, lr, train_loader, val_loader,
                      weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam, lr_sched=None, start_factor:float = 1.0, end_factor:float = 1e-4, steps: int = 4, gamma: float = 0.99999, weights: list = [0.1,0.1,0.3, 1], encoder_forcing: bool = True):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento
        onecycle = False
        linear = False
        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        #Learning rate scheduler
        if lr_sched is not None:    
            try:
                sched = lr_sched(optimizer, lr, epochs=epochs,steps_per_epoch=len(train_loader))
                onecycle = True
            except TypeError:
                try:
                    sched = lr_sched(optimizer, start_factor = start_factor, end_factor=end_factor, total_iters = epochs)
                    linear = True
                except TypeError:
                    sched = lr_sched(optimizer, step_size = round(epochs/steps), gamma = gamma)
                    linear = True
        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            if lr_sched is not None:
                lrs = []
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch, weights, encoder_forcing)
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
                #sched step
                if onecycle:
                    lrs.append(get_lr(optimizer))
                    sched.step()
            if linear:
                lrs.append(get_lr(optimizer))
                sched.step()
            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            if lr_sched is not None:
                result['lrs'] = lrs
                self.epoch_end_one_cycle(epoch, result) #imprimir en pantalla el seguimiento
            else:
                self.epoch_end(epoch, result)

            history.append(result) # añadir a la lista el diccionario de resultados
        return history
    


class LSTMDenoisingAutoEncoder(DAE):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True, num_heads = None, architecture = (20,20,20)):
        super(LSTMDenoisingAutoEncoder, self).__init__()
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



        
        

