from torch.autograd import Variable
import torch.functional as F
import torch.nn as nn
import torch

def accuracy(outputs, labels): ## Calcular la precisión
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
class MultiClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generar predicciones
        loss = F.cross_entropy(out, labels) # Calcular el costo
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generar predicciones
        loss = F.cross_entropy(out, labels)   # Calcular el costo
        acc = accuracy(out, labels)           # Calcular la precisión
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Sacar el valor expectado de todo el conjunto de precisiones
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
    
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

def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

class DeepNeuralNetwork(MultiClassificationBase):
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate

class DeepVanillaRNN(MultiClassificationBase):
    def __init__(self, hidden_size, input_size, output_size, batch_size, mlp_architecture, hidden_state = None):
        super(DeepVanillaRNN, self).__init__()
        if hidden_state is None:
            self.hidden_state = Variable(torch.randn(1, batch_size, hidden_size))
        else:
            self.hidden_state = hidden_state
        self.hidden_mlp = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.input_mlp = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        self.output_mlp = DeepNeuralNetwork(hidden_size, output_size, mlp_architecture)
    def forward(self,x):
        a_t = self.hidden_mlp(self.hidden_state) + self.input_mlp(x)
        self.hidden_state = torch.tanh(a_t)
        output = self.output_mlp(self.hidden_state)
        return output

class DeepLSTM(MultiClassificationBase):
    def __init__(self, hidden_size, input_size, output_size, batch_size, mlp_architecture, hidden_state = None, cell_state = None):
        super(DeepLSTM, self).__init__()
        #Forget gate
        self.F_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.F_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        #Input gate
        self.I_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.I_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        #Ouput gate
        self.O_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.O_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        #Input node
        self.C_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.C_hat_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)        
        #c and H
        if hidden_state is None:
            self.H = Variable(torch.randn(1, batch_size, hidden_size))
        else:
            self.H = hidden_state
        if cell_state is None:
            self.C = Variable(torch.randn(1, batch_size, hidden_size))
        else:
            self.C = cell_state
        #Output mlp
        self.H_h = DeepNeuralNetwork(hidden_size, output_size, mlp_architecture)
    def forward(self,x):
        self.a_F = self.F_h(self.H) + self.F_x(x)
        self.F = nn.Sigmoid(self.a_F)
        self.a_I = self.I_h(self.H) + self.I_x(x)
        self.I = nn.Sigmoid(self.a_I)
        self.a_O = self.O_h(self.H) + self.O_x(x)
        self.O = nn.Sigmoid(self.a_O)
        self.a_C_hat = self.C_hat_h(self.H) + self.C_hat_x(x)
        self.C_hat = nn.tanh(self.a_C_hat)
        self.C = self.F*self.C + self.I*self.C_hat
        self.H = self.O*nn.tanh(self.C)
        output = self.H_h(self.H)
        return output, self.H, self.C

class DeepGRU(MultiClassificationBase):
    def __init__(self, hidden_size, input_size, output_size, batch_size, mlp_architecture, hidden_state = None):
        super(DeepGRU, self).__init__()
        #Update gate
        self.Z_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.Z_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        #Reset gate
        self.R_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.R_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        #Possible hidden state
        self.H_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, mlp_architecture)
        self.H_hat_x = DeepNeuralNetwork(input_size, hidden_size, mlp_architecture)
        #Hidden
        self.H_h = DeepNeuralNetwork(hidden_size, output_size, mlp_architecture)
        if hidden_state is None:
            self.H = Variable(torch.randn(1, batch_size, hidden_size))
        else:
            self.H = hidden_state
    def forward(self, x):
        self.Z = nn.Sigmoid(self.Z_h(self.H)+self.Z_x(x))
        self.R = nn.Sigmoid(self.R_h(self.H)+self.R_x(x))
        self.H_hat = nn.tanh(self.H_hat_h(self.H*self.R)+self.H_hat_x(x))
        self.H = self.H*self.Z + (torch.ones_like(self.Z)-self.Z)*self.H_hat
        output = self.H_h(self.H)
        return output, self.H

class DeepComplexRNN(MultiClassificationBase):
    def __init__(self, hidden_size, input_size, output_size, batch_size, architecture, models):
        self.models = models
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.architecture = architecture
        self.H = Variable(torch.randn(1, batch_size, hidden_size))
        self.C = Variable(torch.randn(1, batch_size, hidden_size))        
    def forward(self, x):
        for model in self.models:
            if type(model)==DeepLSTM:
                model = model(self.hidden_size, self.input_size, self.output_size, self.batch_size, self.architecture, self.H, self.C)
                output, self.H, self.C = model(x)
            else:
                model = model(self.hidden_size, self.input_size, self.output_size, self.batch_size, self.architecture, self.H)
                output, self.H = model(x)
        return output